from .utils import *
from .process_web_page import parse_web_page
import requests
import pandas as pd
import func_timeout
import trafilatura
from ddgs import DDGS
import subprocess
from htmldate import find_date
from mediawiki import MediaWiki


# zip -r websearch_cache.zip results/websearch/queries results/webcontent/metadata_text_table results/webcontent/html
SEARCH_CACHE_DIR = PROJECT_FOLDER/"results/websearch/queries/" # +{query}.json
WEB_CACHE_DIR = PROJECT_FOLDER/"results/webcontent/metadata_text_table/" # +{normalize_str(url)}.txt
WEB_HTML_CACHE_DIR = PROJECT_FOLDER/"results/webcontent/html/" # +{normalize_str(url)}.html
os.makedirs(SEARCH_CACHE_DIR, exist_ok=True)
os.makedirs(WEB_CACHE_DIR, exist_ok=True)
os.makedirs(WEB_HTML_CACHE_DIR, exist_ok=True)
ddgs_search = DDGS()
wikipedia = MediaWiki()


# mapping domain to its attributes in terms of reliability
# Ref: https://en.wikipedia.org/wiki/Sponsored_top-level_domain
NEWS_AGENCY_HOSTNAME_PROPERTY = {
    ".gov": "government, official, reliable",
    ".edu": "education, official, reliable",
    "wikipedia.org": "encyclopedia, objective, normally up-to-date, community-driven, reliable with citations and references",
    "forbes.com": "news, reputable",
    "reuters.com": "news, reputable",
    "nytimes.com": "news, reputable",
    "cnn.com": "news, reputable but with some bias",
    "bbc.com": "news, reputable but with some bias",
    "foxnews.com": "news, less reputable, with more bias",
    "linkedin.com": "social media, opinion-based, not always reliable",
    "x.com": "social media, opinion-based, not always reliable",
    "facebook.com": "social media, opinion-based, not always reliable",
    "others": "unknown reliability, need to check the source and content carefully"
}


class WebRetriever(Retriever):
    """
    A retriever that performs web search and fetches relevant web articles. By default, return top 10 results.

    Searching:
        - Uses DDGS/OpenSerp/MediaWiki packages (with google, duckduckgo, wikipedia engines, etc.)
        - Return a list of dict{"title", "href", "body"}
    Fetching:
        - Fetching content of URLs in HTML format by DDGS (can bypass protected websites)
        - Extracting semantic units (tables and remaining HTML text) by a custom function
        - Convert HTML text to clean Markdown by Trafilatura
        - Return metadata, text, tables for each URL.
    Formatting:
        - Convert the extracted content into a unified format (e.g., Match object with Paragraph and Table)
    """

    def __init__(self, **kwargs):
        super().__init__(type_="web", **kwargs)

    def search(self, query: str, top_k: int = 10,
               backend: str="duckduckgo", webcontent_format="semantic_units")-> list[Match]:
        """
        Web search using DDGS/OpenSerp/MediaWiki packages (with google, duckduckgo, wikipedia engines, etc.) and fetch relevant web articles.

        Args:
            query: The search query string.
            top_k: The number of top search results to return. Default is 10.
            backend: The search engine backend to use. Comma separated string, e.g., "duckduckgo,google,wikipedia".
            webcontent_format: The search results' format to chunk/process web content into. Either
                - "semantic_units" (default, chunk into semantic units like tables and paragraphs) or 
                - "full_page_markdown" (no chunking, return the whole text as it is).

        Returns:
            A list of `Match` objects containing the retrieved paragraphs and tables from the web search results.
        """
        self.update_counter()
            
        # Retrieve/Save search results from/to cache
        basename = f"{normalize_str(query)}-{top_k}-{backend.replace(',','-')}.txt"
        filepath = os.path.join(SEARCH_CACHE_DIR, basename)
        search_results = []
        if os.path.exists(filepath):
            print_debug(f"Loading cached search results \n\t from '{filepath}' for \n\t query '{query}'")
            with open(filepath, "r", encoding="utf-8") as f:
                search_results = json.load(f)

        if is_search_results_empty(search_results):
            try:
                wiki_search_results = _search_wikipedia(query=query, top_k=top_k) if "wikipedia" in backend else []
                backend = backend.replace(",wikipedia", "")
                if backend == "auto":
                    search_results = _search_ddgs(query=query, max_results=top_k) # ddgs
                else:
                    try:
                        search_results = _search_openserp(query=query, max_results=top_k, backend=backend) # openserp
                    except Exception as e: # fallback to ddgs
                        print_debug(f"OpenSerp search failed for query '{query}' with error {e}, fallback to DDGS search.")
                        search_results = _search_ddgs(query=query, max_results=top_k, backend=backend)
                    urls = [r.get("href", "") for r in search_results]
                    for a in wiki_search_results:
                        if a["href"] not in urls:
                            search_results.append(a)
                print_debug(f"Caching search results \n\t to '{filepath}' for \n\t query '{query}'")
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(search_results, f)
            except Exception as e:
                print_debug(f"Web search failed for query '{query}' with error {e}, returning empty results.")
                search_results = []
            except func_timeout.FunctionTimedOut:
                print_debug(f"Web search timed out for query '{query}', returning empty results.")
                search_results = []
        
        # Load/fetch articles from/to cache, then convert to Match format
        # TODO: process PDF files, e.g., "https://web.mit.edu/hackl/www/lab/turkshop/slides/regex-cheatsheet.pdf"
        matches = []
        for r in search_results:
            title = r.get("title", "")
            url = r.get("href")
            print_debug(f"Processing {title}: {url}")
            if not url or re.search(r"pdf|xls|doc", url.lower()):   # TODO: Implement office files extraction.
                continue
            basename = f"{normalize_str(title)}.md"
            filepath = os.path.join(WEB_CACHE_DIR, basename)
            if os.path.exists(filepath):
                metadata, text, tables = _load_webpage(filepath, url)
            else:
                try:
                    # Try to use trafilatura to fetch content, as it is cleaner. But it's less robust.
                    metadata, text, tables = _crawl_web(url, title=title, with_metadata=True)
                    if not text:
                        # Fallback to ddgs extract, which is noiser but more robust
                        # Extract rich text (headers/lists, no hyperlinks).
                        text = ddgs_search.extract(url, fmt="text_rich").get("content", "")
                    if not metadata:
                        metadata = {"url": url, "title": title}
                    _save_webpage(filepath, metadata, text, tables)
                except BaseException as e:
                    print_debug(f"Failed to fetch content for url '{url}' with error {e}, using empty content.")
                    metadata = {"url": url, "title": title}
                    text = ""
                    tables = []
                except func_timeout.FunctionTimedOut:
                    print_debug(f"Web crawling timed out for url '{url}', using empty content.")
                    metadata = {"url": url, "title": title}
                    text = ""
                    tables = []

            metadata["hostname"] = url.split('://', 1)[-1].split('/', 1)[0]
            if tables:
                for table in tables:
                    if is_table_not_useful(table):
                        continue

                    # df = pd.DataFrame(table.get("dataframe", []))
                    # df_markdown = re.sub(r"[ ]+", " ", df.to_markdown(index=False)) if not df.empty else ""
                    headings = [h["text"] for h in table.get("headings", [])]
                    if len(headings) == 0 or headings[0].lower() not in title.lower():
                        headings.insert(0, title)
                    caption = table.get("table_metadata", {}).get("caption", "")
                    if caption and caption not in headings:
                        headings.append(caption)
                    anchor = "Table (in JSONL format) regarding " + " / ".join(headings)
                    df = table.get("dataframe", [])
                    dataframe_jsonl = "\n".join([json.dumps(row) for row in df])
                    content = f"{anchor}\n```jsonl\n{dataframe_jsonl}\n```\n" # f"{anchor}\n{df_markdown}"

                    # treat it as text if the table is too big or dense or the webcontent_format is full_page_markdown
                    if webcontent_format == "full_page_markdown" or \
                        len(df) == 0 or len(df[0]) == 0: # or \
                        # (len(dataframe_jsonl.split()) // (len(df) * len(df[0]) + 1)) > 50:
                        text += "\n\n" + dataframe_jsonl
                        continue

                    table_paragraph = Table(
                        filepath=filepath,
                        metadata=metadata,
                        preview=content[:200],
                        content=content,
                        **table
                    )
                    match = Match(
                        search_type="web",
                        paragraph=table_paragraph,
                        # preview=table_paragraph.preview,
                        retriever=self
                    )
                    matches.append(match)
            
            if text:
                text = clean_up_text(metadata, text)
                paragraph = Paragraph(
                    filepath=filepath,
                    metadata=metadata,
                    preview=r.get("body", ""),
                    content=text
                )
                match = Match(
                    search_type="web",
                    paragraph=paragraph,
                    # preview=paragraph.preview,
                    retriever=self
                )
                matches.append(match)

        return matches

    def __str__(self):
        return (
            "Web search"
            # f"Web search retriever using DDGS package with google and wikipedia engines, etc. "
            # f"Search results and fetched web content are cached in folders `{SEARCH_CACHE_DIR}` and `{WEB_CACHE_DIR}` respectively."
        )


@func_timeout.func_set_timeout(20)
def _search_wikipedia(query, top_k=10):
    page_titles = wikipedia.search(query, results=top_k)
    results = [] # in {"title", "href", "body"} format for compatibility with DDGS search results.
    for p in page_titles:
        page = wikipedia.page(p)
        results.append({
            "title": page.title,
            "href": page.url,
            "body": page.summary,
        })
    return results


@func_timeout.func_set_timeout(30)
def _search_ddgs(**kwargs):
    time.sleep(5) # avoid too many requests
    return ddgs_search.text(**kwargs)


# OpenSerp https://github.com/karust/openserp
# curl command: curl "http://localhost:7000/mega/search?text={query}&engines=google,duckduckgo&limit={count}"
@func_timeout.func_set_timeout(60)
def _search_openserp(query, max_results=10, backend="duckduckgo"):
    """
    Web search using OpenSerp package
    Return: List of dict_keys(['title', 'href', 'body', ..])
    """
    port = os.getenv("OPENSERP_PORT", "7000")
    url = f"http://localhost:{port}/mega/search"
    params = {"text": query, "engines": backend, "limit": max_results}
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    search_results = resp.json()["results"]
    formatted_search_results = []
    for r in search_results:
        formatted_search_results.append({
            "title": r.get("title"),
            "href": r.get("url"),
            "body": r.get("snippet")
        })
    return formatted_search_results


@func_timeout.func_set_timeout(20)
def _crawl_web(url, title, with_metadata=False):
    """
    Crawl the webpage using DDGS and Trafilatura package, return the text content.
    - Using DDGS to fetch html page robustly (bypassing protected pages), and 
    - Process the html with trafilatura to get cleaner text.
    
    Maximize recall by parsing in Markdown format.
    Docs:
    - https://github.com/deedy5/ddgs?tab=readme-ov-file#6-extract
    - https://trafilatura.readthedocs.io/en/latest/quickstart.html
    CLI:
    - ddgs extract -u "https://www.forbes.com/lists/athletes/" -f text
    - trafilatura -u https://2024.emnlp.org/ --markdown

    NOTE: Sections, paragraphs or tables are well separated with double new line char '\n\n' 
    """
    # md = ddgs_search.extract(url, fmt="text_rich")["content"] # Markdown w/o URLs. Not follow standard Markdown requirements.
    basename = f"{normalize_str(title)}.html"
    html_filepath = os.path.join(WEB_HTML_CACHE_DIR, basename)
    if os.path.exists(html_filepath):
        print_debug(f"Loading cached HTML content \n\t from '{html_filepath}' for \n\t url '{url}'")
        with open(html_filepath, "r", encoding="utf-8") as f:
            html = f.read()
    else:
        try:
            # html = ddgs_search.extract(url, fmt="text")["content"] # alternative: trafilatura.fetch_url(url).
            # use system run to avoid Rust panic error
            subprocess.getoutput(f"ddgs extract -u \"{url}\" -f text > {html_filepath}")
            with open(html_filepath, "r", encoding="utf-8") as f:
                html = f.read()
        except Exception as e:
            html = trafilatura.fetch_url(url)
            with open(html_filepath, "w", encoding="utf-8") as f:
                f.write(html)
    if not html:
        raise ValueError(f"Failed to fetch HTML content for url '{url}' using both DDGS and Trafilatura.")

    # md = trafilatura.extract(html, output_format="markdown", include_comments=False) # Follow standard Markdown requirements.
    tables_and_text_dict = parse_web_page(html) # {"tables": tables, "text": remaining_text,}
    # remove emojis and other non-utf-8 characters that may cause issues in trafilatura or LLM inference.
    html = html.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = tables_and_text_dict.get("text", "")
    tables = tables_and_text_dict.get("tables", [])
    if with_metadata:
        metadata = trafilatura.extract_metadata(html).as_dict() # various important metadata, e.g., publication date,
        hostname = url.split('://', 1)[-1].split('/', 1)[0]
        metadata["hostname"] = hostname
        for k in ["body", "commentsbody", "raw_text", "text"]:
            metadata.pop(k, None)
        metadata["last_modification_date"] = find_date(html)
        metadata["url"] = url
        metadata["title"] = title
        return metadata, text, tables
    else:
        return text, tables


def _load_webpage(filepath, url):
    print_debug(f"Loading cached web content \n\t from '{filepath}' for \n\t url '{url}'")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    metadata_str, text, tables = re.split(rf"{METADATA_TEXT_SEP}|{TEXT_TABLE_SEP}", content, maxsplit=2)
    metadata = json.loads(metadata_str)
    tables = json.loads(tables.strip().lstrip("```python\n").rstrip("\n```"))
    return metadata, text, tables


def _save_webpage(filepath, metadata, text, tables=[]):
    print_debug(f"Caching web content \n\t to '{filepath}' for \n\t url '{metadata['url']}'")
    content = json.dumps(metadata, indent=2) + METADATA_TEXT_SEP + text + \
        TEXT_TABLE_SEP + "```python\n" + json.dumps(tables, indent=2) + "\n```"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def clean_up_text(metadata, text):
    if "wikipedia.org" in metadata["url"]:
        # For Wikipedia articles, remove the "References", "External links", "See also" sections, etc. to improve precision.
        text = re.split(r"# References|# External links|# See also|# Bibliography", text)[0]
    return text


def is_table_not_useful(table):
    # determine if the table is not useful for QA based on its classes.
    # For example, tables with "navbox" or "sidebar" classes in Wikipedia are usually not useful for QA.
    ignored_classes = ['navbox', 'sidebar']
    classes = table.get("table_metadata", {}).get("class", [])
    if not classes:
        classes = []
    table_classes = "-----".join(classes)
    if any([_ in table_classes for _ in ignored_classes]):
        print_debug(f"  Detected a table of ignored class. Skipping for now.")
        return True
    return False


def is_search_results_empty(search_results: list):
    # determine if the whole search results is empty.
    if not search_results:
        return True
    try:
        return not any([r.get("href", "") for r in search_results])
    except Exception as e:
        return True


def normalize_str(url):
    """
    Normalize the url to a valid filename, by replacing special characters with `-`.
    Also chunk the string up to 246 chars, as linux file basename length limit = 255 bytes
    """
    return re.sub(r'[^a-zA-Z0-9]', '-', url)[:246]


def demo():
    results = ddgs_search.text(
        query="Python Programming", 
        max_results=10,
        backend="google,duckduckgo,wikipedia",
    )
    articles = []
    for r in results:
        articles.append(ddgs_search.extract(r.get("href"))["content"][:100])
    print_debug(results)
    print_debug(articles)


def test_search_web():
    query = "number of episodes per season of The Simpsons"
    retriever = WebRetriever()
    print_debug(retriever)

    results = retriever.search(query, top_k=5)
    print_debug(f"Search Type: `web`")
    for match in results:
        print_debug(match)
        if match.paragraph is not None:
            print_debug(f"File: {match.paragraph.filepath}")
            print_debug(f"URL: {match.paragraph.metadata['url']}")
            print_debug(f"Preview: {match.paragraph.preview}")
        print_debug("-" * 40)


def test_crawl_web():
    url = "https://en.wikipedia.org/wiki/List_of_episodes_of_The_Simpsons"
    metadata, text, tables = _crawl_web(url, "List_of_episodes_of_The_Simpsons", with_metadata=True)
    print_debug(json.dumps({"metadata": metadata, "text": text, "tables": tables}, indent=2))


if __name__ == "__main__":
    # demo()
    # test_crawl_web()
    test_search_web()
