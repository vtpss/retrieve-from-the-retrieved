import os, re, glob, json, time
from tqdm import tqdm
from typing import Literal, Union
from dataclasses import dataclass
# from rich import print

# __all__ = ["_chunks_by_sliding_window", "_build_preview", "fuzz"]
# if we name function with leading underscore, it won't be imported when do "from utils import *",
# but it can still be imported by "from utils import _chunks_by_sliding_window". 
# So it's a soft convention to indicate internal use only, but not enforced by Python.

from pathlib import Path
PROJECT_FOLDER = Path(__file__).parent.parent

METADATA_TEXT_SEP = "\n\n--METADATA-TEXT-SEP--\n\n"
TEXT_TABLE_SEP = "\n\n--TEXT-TABLE-SEP--\n\n"

import tiktoken
tiktoken_encoder = tiktoken.get_encoding("o200k_base")
count_tokens = lambda text: len(tiktoken_encoder.encode("".join(text), disallowed_special=()))

def print_debug(*args, **kwargs):
    if int(os.getenv("DEBUG", "0")):
        print(*args, **kwargs)

print_hyphen = lambda n: print_debug("\n" + "-"*n + "\n")


@dataclass
class Paragraph:    # Span of text, but use `Paragraph` as the name to emphasize on the semantic completeness
    filepath: str
    metadata: dict = None
    preview: str = ""   # a short preview of the paragraph, e.g., summary, or matched text if it's a search result,
                        # or the first 200 chars if it's just a chunk of document.
    content: str = ""
    original: str = ""  # in case the content is modified, keep the original content for reference.
    prev_paragraph = None   # link to the previous paragraph, for better managing the evidence chain.
    next_paragraph = None   # link to the next paragraph, for better managing the evidence chain.
    # neighbor_paragraph = None # Some paragraph that links to this paragraph to form a multi-hop knowledge

    # def __str__(self):
    #     return "Paragraph(filepath='{}', metadata='{}', preview='{}', content='{}')".format(
    #         self.filepath, self.metadata, self.preview, self.content[:200]
    #     )
    
    def __rich_repr__(self):
        yield "filepath", self.filepath
        yield "metadata", self.metadata
        yield "preview", self.preview
        yield "content", self.content # [:200] + "..."

    def __eq__(self, value):
        if not isinstance(value, Paragraph):
            return False
        return self.filepath == value.filepath and self.content == value.content
    
    def __hash__(self):
        return hash((self.filepath, self.content))


@dataclass
class Table(Paragraph):
    dataframe: list[dict] = None    # df in `records`` format
    headings: list[dict] = None     # list of {"level": int, "text": str} for the heading hierarchy above the table
    table_metadata: dict = None     # e.g. {"id": str, "class": str, "caption": str}


@dataclass
class Match:
    search_type: Literal["regex", "fuzzy", "sparse", "dense"]   # the search method that produced this match
    paragraph: Paragraph
    preview: str = ""           # a short preview of the matched content, e.g. 20 words before and after the matched query span
    matched_text: str = None    # the exact matched text span
    matched_score: int = None   # the matched score, e.g. fuzzy score, regex match groups, BM25 score, embedding similarity, etc. 
    chunk_id: int | None = None
    retriever: "Retriever" = None   # the retriever that produced this match, used for managing retriever pool in the search loop

    # def __str__(self):
    #     return f"Match(search_type='{self.search_type}', paragraph={str(self.paragraph)}, preview='{self.preview}', matched_text='{self.matched_text}', matched_score={self.matched_score}, chunk_id={self.chunk_id})"
    
    def __rich_repr__(self):
        yield "search_type", self.search_type
        yield "paragraph", self.paragraph
        yield "preview", self.preview
        yield "matched_text", self.matched_text
        yield "matched_score", self.matched_score
        yield "chunk_id", self.chunk_id


# Template https://github.com/texttron/BrowseComp-Plus/blob/main/searcher/searchers/custom_searcher.py
class Retriever:
    """
    A retriever interface that can be implemented by different search methods, 
    e.g. BM25, dense retrieval, web search, etc.
    """
    counter = {
        "sparse": 0,
        "dense": 0,
        "web": 0,
        "other": 0
    }

    def __init__(self, type_: str, paragraphs: list[Paragraph]=None, folder: str=None, **kwargs):
        self.type = type_ # Literal["regex", "fuzzy", "sparse", "dense", "web"]
        self.paragraphs = None
        self.folder = None
        if type_ == "web":
            pass
        elif paragraphs:
            self.paragraphs = paragraphs
        else:
            self.folder = folder
            paragraph_sep = kwargs["paragraph_sep"] if "paragraph_sep" in kwargs else "\n\n"
            self.paragraphs = load_files_in_folder_as_paragraphs(folder, paragraph_sep=paragraph_sep) if folder else []
            if not self.paragraphs:
                raise ValueError("Either non-empty folder or list of paragraphs must be provided to initialize Retriever.")
            # print(f"Initialized {self.type} retriever with {len(self.paragraphs)} paragraphs from folder `{folder}`.")

    def search(self, query: str, top_k: int = -1):
        """
        Retrieve/search relevant passages/articles/documents for the query, 
        Returning a list of Match objects ordered by score descending.
        """
        raise NotImplementedError("Retriever subclasses must implement the search method.")    

    def update_counter(self):
        """Update the counter for this retriever type."""
        if self.type in Retriever.counter:
            Retriever.counter[self.type] += 1
        else:
            Retriever.counter["other"] += 1

    def __str__(self):
        """
        Return a string description of the retriever, 
        e.g. its type and capabilities, used for the agent to decide which retriever to use.
        """
        raise NotImplementedError("Retriever subclasses must implement the __str__ method.")
    
    def __rich_repr__(self):
        self.__str__()


#---# Helpers for paragraph management

def get_paragraph_content(paragraph: Paragraph) -> str:
    """Get the content of a paragraph, either from the `content` field or by reading the file."""
    if paragraph.content:
        return paragraph.content
    if paragraph.filepath and os.path.isfile(paragraph.filepath):
        try:
            paragraph.content = open(paragraph.filepath, encoding="utf-8", errors="ignore").read()
            paragraph.preview = paragraph.content[:200] if not paragraph.preview else paragraph.preview
            return paragraph.content
        except OSError:
            return ""
    return ""


def chunk_long_paragraph_by_max_chars(chunk: str, max_char=1024, regex_clean_cut=r"([.!?])"):
    """
    Chunk a long paragraph by max number of characters, e.g. 1024 characters.
    Attempt to chunk at the end of sentence.
    """
    if len(chunk) <= max_char:
        return [chunk]
    else:
        chunks = []
        while len(chunk) > max_char:
            current_chunk = chunk[:max_char]
            # Try to find a clean cut point
            last_period_pos = re.search(regex_clean_cut, current_chunk[::-1])
            if last_period_pos:
                current_chunk = current_chunk[:max_char-last_period_pos.start()]
            else:
                current_chunk = current_chunk.rsplit(" ", 1)[0]  # if no period, try to split at the last space
            chunk = chunk[len(current_chunk):].strip()
            chunks.append(current_chunk)
        chunks.append(chunk)
        return chunks


def chunk_paragraph(paragraph: Paragraph, paragraph_sep: str="\n\n") -> list[Paragraph]:
    """
    Chunk a Paragraph into shorter Paragraph(s).
    Shorter paragraphs should inherit filepath and metadata of the original paragraph, and 
        have their own preview (first 200 chars or preview of the original paragraph if the preview is in the content).
    """
    chunked_paragraphs = []
    content = get_paragraph_content(paragraph)
    if not content:
        return chunked_paragraphs
    if paragraph_sep is None:
        chunked_paragraphs.append(paragraph)
    else:
        chunks = [c for _ in re.split(paragraph_sep, content) for c in chunk_long_paragraph_by_max_chars(_)]
        previous_chunk = None
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk:  # only add non-empty paragraphs
                preview = paragraph.preview if (paragraph.preview and paragraph.preview in chunk) else chunk[:200]
                current_chunk = Paragraph(
                    filepath=paragraph.filepath,
                    metadata=paragraph.metadata,
                    preview=preview,
                    content=chunk
                )
                if previous_chunk is not None:
                    previous_chunk.next_paragraph = current_chunk
                    current_chunk.prev_paragraph = previous_chunk
                chunked_paragraphs.append(current_chunk)
                previous_chunk = current_chunk
        
        if paragraph.prev_paragraph:
            chunked_paragraphs[0].prev_paragraph = paragraph.prev_paragraph
            paragraph.prev_paragraph.next_paragraph = chunked_paragraphs[0]
        if paragraph.next_paragraph:
            chunked_paragraphs[-1].next_paragraph = paragraph.next_paragraph
            paragraph.next_paragraph.prev_paragraph = chunked_paragraphs[-1]

    return chunked_paragraphs

'''
def span_paragraphs(paragraphs: list[Paragraph], window_size=2, paragraph_sep: str="\n\n") -> list[Paragraph]:
    """
    Simply get the span of each paragraph in the given list of Paragraphs
        by merging the neighboring paragraphs within a window size into a single Paragraph using the linked list.
    """
    if not paragraphs:
        return []

    # sort paragraphs by filepath and their order in the file (using prev/next pointer)
    spanned_paragraphs = []

    for p in paragraphs:
        current_span = [p]
        # expand the span to the prev paragraphs within the window size
        current = p
        count = window_size
        while count > 0:
            if current.prev_paragraph:
                current = current.prev_paragraph
                current_span.insert(0, current)
                count -= 1
            else:
                break
        # expand the span to the next paragraphs within the window size
        current = p
        count = window_size
        while count > 0:
            if current.next_paragraph:
                current = current.next_paragraph
                current_span.append(current)
                count -= 1
            else:
                break

        # merge the span into one paragraph
        if len(current_span) > 1:
            merged_content = paragraph_sep.join([get_paragraph_content(par) for par in current_span])
            merged_preview = "...\n".join([par.preview for par in current_span if par.preview])
            new_p = Paragraph(
                filepath=p.filepath,
                metadata=p.metadata,
                preview=merged_preview,
                content=merged_content,
            )
            spanned_paragraphs.append(new_p)
        else:
            spanned_paragraphs.append(p)
    return spanned_paragraphs

'''
def span_paragraphs(paragraphs: list[Paragraph], window_size=2, paragraph_sep: str="\n\n") -> list[Paragraph]:
    """
    Given a list of Paragraphs in different files.
    For Paragraphs in the sample files, merge the span of paragraphs within a window size into a single Paragraph using the linked list.
        If two given paragraphs share window span, their will be merged into one Paragraph.
    Also update the pointer of prev and next paragraph properly
    """
    if not paragraphs:
        return None

    # sort paragraphs by filepath and their order in the file (using prev/next pointer)
    paragraphs.sort(key=lambda p: (p.filepath, get_paragraph_content(p)))
    spanned_paragraphs = []
    visited = []

    for p in paragraphs:
        if p in visited:
            continue
        current_span = [p]

        # expand the span to the prev paragraphs within the window size
        current = p
        count = window_size
        while count > 0:
            if current.prev_paragraph:
                current = current.prev_paragraph
                visited.append(current)
                current_span.insert(0, current)
                count -= 1
                # reset the count if the prev paragraph is also in the given paragraphs
                if current in paragraphs:
                    count = window_size
            else:
                break

        # expand the span to the next paragraphs within the window size
        current = p
        count = window_size
        while count > 0:
            if current.next_paragraph:
                current = current.next_paragraph
                visited.append(current)
                current_span.append(current)
                count -= 1
                # reset the count if the next paragraph is also in the given paragraphs
                if current in paragraphs:
                    count = window_size
            else:
                break

        # merge the span into one paragraph
        if len(current_span) > 1:
            merged_content = paragraph_sep.join([get_paragraph_content(par) for par in current_span])
            merged_preview = "...\n".join([par.preview for par in current_span if par.preview])
            new_p = Paragraph(
                filepath=p.filepath,
                metadata=p.metadata,
                preview=merged_preview,
                content=merged_content,
            )
            # update the pointer of prev and next paragraph
            if current_span[0].prev_paragraph:
                current_span[0].prev_paragraph.next_paragraph = new_p
                new_p.prev_paragraph = current_span[0].prev_paragraph
            if current_span[-1].next_paragraph:
                current_span[-1].next_paragraph.prev_paragraph = new_p
                new_p.next_paragraph = current_span[-1].next_paragraph
            spanned_paragraphs.append(new_p)
        else:
            spanned_paragraphs.append(p)

    spanned_paragraphs = [s for s in spanned_paragraphs if s not in visited]
    return spanned_paragraphs


def load_files_in_folder_as_paragraphs(
    folder: str,
    preview_type: Literal["firstchar", "filename"] = "firstchar",
    paragraph_sep: Literal["\n\n", "\n", None] = "\n\n"
) -> list[Paragraph]:
    """
    Get a list of Paragraphs for files in the folder.
    Each file is split by `paragraph_sep` and each non-empty chunk becomes a `Paragraph`. 
    """
    paragraphs: list[Paragraph] = []
    all_files = glob.glob(os.path.join(folder, "**"), recursive=True)
    for file in tqdm(all_files, desc=f"Loading files in folder `{folder}` as paragraphs", total=len(all_files)):
        if not os.path.isfile(file):
            continue

        if paragraph_sep is None:
            # treat the whole file as one paragraph
            paragraphs.append(Paragraph(
                filepath=file,
                metadata={},
                preview="" # fin.read(200) if preview_type == "firstchar" else os.path.basename(file)
            ))
        elif paragraph_sep == METADATA_TEXT_SEP:
            # load file as web content format with metadata and text separated by METADATA_TEXT_SEP
            content = open(file, encoding="utf-8", errors="ignore").read()
            metadata_str, text = content.split(METADATA_TEXT_SEP, 1)
            paragraphs.append(Paragraph(
                filepath=file,
                metadata=json.loads(metadata_str),
                preview=text[:200] if preview_type == "firstchar" else os.path.basename(file),
                content=text
            ))
        else:
            fin = open(file, encoding="utf-8", errors="ignore")
            content = fin.read()
            chunks = re.split(paragraph_sep, content)
            previous_chunk = None
            for chunk in chunks:
                chunk = chunk.strip()
                if chunk:  # only add non-empty paragraphs
                    current_chunk = Paragraph(
                        filepath=file,
                        metadata={},
                        preview=chunk[:200],
                        content=chunk
                    )
                    if previous_chunk is not None:
                        previous_chunk.next_paragraph = current_chunk
                        current_chunk.prev_paragraph = previous_chunk
                    paragraphs.append(current_chunk)
                    previous_chunk = current_chunk

    return paragraphs


def str_enumerate(items):
    return "\n".join([f"{i}. {item}" for i, item in enumerate(items)])


def paragraph_to_text(paragraph: Paragraph, with_metadata=True, agency_attribute_map=None):
    if with_metadata:
        hostname = str(paragraph.metadata.get("hostname", "")).lower()
        attribute = agency_attribute_map["others"]
        for na in agency_attribute_map.keys():
            if na in hostname:
                attribute = agency_attribute_map[na]
                break
        return (
            f"> Source: {hostname} ({attribute})\n" \
            f"> Last Modification Date: {paragraph.metadata.get('last_modification_date', 'N/A')}\n" \
            f"> Title: {paragraph.metadata.get('title', 'N/A')}\n" \
            f"> Description: {paragraph.metadata.get('description', 'N/A')}\n" \
            f"\n```\n{paragraph.content}\n```"
        )
    else:
        return paragraph.content


#---# Other helpers

def chunks_by_sliding_window(content, window: int = 3, slide: int = 2) -> list[tuple[int, list[str]]]:
    """Create sliding window chunks over content, returning (start_line_no, pieces)."""
    pieces = content.splitlines()
    chunks = []
    i = 0
    while i + window <= len(pieces):
        chunks.append((i + 1, pieces[i:i + window]))  # 1-indexed line numbers
        i += slide
    return chunks


def build_preview(pieces: list[str], query_words: list[str]) -> str:
    """Build preview: matched query span + 10 words before and after."""
    chunk_text = " ".join(pieces)
    chunk_lower = chunk_text.lower()
    # Find first occurrence of any query word in chunk
    first_pos = -1
    for w in query_words:
        pos = chunk_lower.find(w)
        if pos != -1 and (first_pos == -1 or pos < first_pos):
            first_pos = pos
    if first_pos == -1:
        return chunk_text[:200]
    # Extract window around match
    words_before = chunk_text[:first_pos].split()
    window_words = chunk_text[first_pos:].split()
    # Take up to 10 words before, the matched span, up to 10 words after
    before = " ".join(words_before[-10:]) if len(words_before) > 10 else " ".join(words_before)
    after = " ".join(window_words[:21])  # ~1 word query + 10 before + 10 after
    return f"...{before} {after}...".strip()


#---# Test cases

def test_rich_pretty_print():
    paragraph = Paragraph(filepath="test.txt", preview=None, metadata=None, content="Test pretty print. "*20)
    print(paragraph)
    match = Match(search_type="regex", paragraph=paragraph, preview="Test pretty print. Test pretty print.", matched_text="pretty", matched_score=90, chunk_id=1)
    print(match)


def test_span_paragraphs():
    # create paragraphs with linked list
    doc1 = [
        Paragraph(filepath="file1", content="Paragraph 1"),
        Paragraph(filepath="file1", content="Paragraph 2"),
        Paragraph(filepath="file1", content="Paragraph 3"),
        Paragraph(filepath="file1", content="Paragraph 4"),
        Paragraph(filepath="file1", content="Paragraph 5"),
        Paragraph(filepath="file1", content="Paragraph 6"),
        Paragraph(filepath="file1", content="Paragraph 7"),
        Paragraph(filepath="file1", content="Paragraph 8"),
        Paragraph(filepath="file1", content="Paragraph 9"),
        Paragraph(filepath="file1", content="Paragraph 10"),
        Paragraph(filepath="file1", content="Paragraph 11"),
        Paragraph(filepath="file1", content="Paragraph 12"),
    ]
    doc2 = [
        Paragraph(filepath="file2", content="Paragraph A"),
        Paragraph(filepath="file2", content="Paragraph B"),
        Paragraph(filepath="file2", content="Paragraph C"),
        Paragraph(filepath="file2", content="Paragraph D"),
        Paragraph(filepath="file2", content="Paragraph E"),
    ]
    for i in range(len(doc1)-1):
        doc1[i].next_paragraph = doc1[i+1]
        doc1[i+1].prev_paragraph = doc1[i]
    for i in range(len(doc2)-1):
        doc2[i].next_paragraph = doc2[i+1]
        doc2[i+1].prev_paragraph = doc2[i]

    paragraphs = [doc1[2], doc1[7], doc1[8], doc1[11], doc2[1]]
    spanned_paragraphs = span_paragraphs(paragraphs, window_size=2)
    for p in spanned_paragraphs:
        print(p)


def test_paragraph_comparison():
    paragraph1 = Paragraph(filepath='/home/quyetdo/.cache/webcontent/urls/List-of-most-viewed-YouTube-videos---Wikipedia.md', preview='Older videos that pre-dated the launch of YouTube in 2005 but were added later to pass a billion views are as follow:...', metadata={'title': 'List of most-viewed YouTube videos - Wikipedia', 'author': None, 'url': 'https://en.wikipedia.org/wiki/List_of_most-viewed_YouTube_videos', 'hostname': 'wikipedia.org', 'description': None, 'sitename': 'Wikimedia Foundation, Inc.', 'date': '2012-11-25', 'categories': [], 'tags': [], 'fingerprint': None, 'id': None, 'license': None, 'comments': None, 'language': None, 'image': None, 'pagetype': 'website', 'filedate': '2026-04-29', 'last_modification_date': '2026-04-28'}, content='Older videos that pre-dated the launch of YouTube in 2005 but were added later to pass a billion views are as follow:\n\n- "November Rain" by Guns N\' Roses became the first video made prior to 2005, YouTube\'s foundation year, to reach this threshold by July 2018.\n[58] - "Numb" by Linkin Park was the first 2000s video predating YouTube to reach 1 billion views in November 2018.\n[59] - "Bohemian Rhapsody" by Queen was the first 1970s video (and pre-1990s video) to reach 1 billion views in July 2019.\n[60] - "Sweet Child o\' Mine" by Guns N\' Roses was the first 1980s video to reach 1 billion views in October 2019.\n[61] - "Zombie" by The Cranberries was the first female sung pre-2005 video to reach 1 billion views in April 2020.\n[62]')
    paragraph2 = Paragraph(filepath='/home/quyetdo/.cache/webcontent/urls/List-of-most-viewed-YouTube-videos---Wikipedia.md', preview='Older videos that pre-dated the launch of YouTube in 2005 but were added later to pass a billion views are as follow:...', metadata={'title': 'List of most-viewed YouTube videos - Wikipedia', 'author': None, 'url': 'https://en.wikipedia.org/wiki/List_of_most-viewed_YouTube_videos', 'hostname': 'wikipedia.org', 'description': None, 'sitename': 'Wikimedia Foundation, Inc.', 'date': '2012-11-25', 'categories': [], 'tags': [], 'fingerprint': None, 'id': None, 'license': None, 'comments': None, 'language': None, 'image': None, 'pagetype': 'website', 'filedate': '2026-04-29', 'last_modification_date': '2026-04-28'}, content='Older videos that pre-dated the launch of YouTube in 2005 but were added later to pass a billion views are as follow:\n\n- "November Rain" by Guns N\' Roses became the first video made prior to 2005, YouTube\'s foundation year, to reach this threshold by July 2018.\n[58] - "Numb" by Linkin Park was the first 2000s video predating YouTube to reach 1 billion views in November 2018.\n[59] - "Bohemian Rhapsody" by Queen was the first 1970s video (and pre-1990s video) to reach 1 billion views in July 2019.\n[60] - "Sweet Child o\' Mine" by Guns N\' Roses was the first 1980s video to reach 1 billion views in October 2019.\n[61] - "Zombie" by The Cranberries was the first female sung pre-2005 video to reach 1 billion views in April 2020.\n[62]')
    print(paragraph1 == paragraph2)  # should be True

    # test with list
    list1 = [paragraph1, paragraph2]
    list2 = [paragraph1]
    for e in list2:
        if e not in list1:
            list1.append(e)
    print(len(list1) == 2)  # should be 2, not 3


if __name__ == "__main__":
    # test pretty print of Paragraph and Match
    test_rich_pretty_print()

    # test span_paragraphs function
    test_span_paragraphs()

    # test paragraph comparison
    test_paragraph_comparison()
