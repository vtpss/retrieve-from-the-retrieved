from .utils import *
import bm25s
import Stemmer  # optional: for stemming
stemmer = Stemmer.Stemmer("english") # Skip stemming for speed


class SparseRetriever(Retriever):
    """
    A retriever that performs sparse search over given paragraphs or given folder.
    
    It builds a BM25+ index from `self.paragraphs` on initialization and uses it to perform sparse retrieval.
    """
    index_folder = os.path.join(PROJECT_FOLDER, "benchmarks/index/sparse")

    def __init__(self, index_name: str, **kwargs):
        super().__init__(type_="sparse", **kwargs)
        self.index_name = index_name
        self.index_path = os.path.join(SparseRetriever.index_folder, index_name)
        self.index = _build_sparse_index(self, index_name)
        # self.index.save(os.path.join(SparseRetriever.index_folder, self.index_name))

    def search(self, query: str, top_k: int = 10, max_total_words: int = 512) -> list[Match]:
        """
        Sparse (BM25+) search on an BM25+ index.

        Args:
            query: The search query string.
            top_k: The number of top results to return. If -1, return as many as possible until reaching the max_total_words limit.
            max_total_words: The maximum total number of words in the retrieved paragraphs when top_k is -1. Only applicable when top_k is -1.

        Returns:
            A list of `Match` objects containing the retrieved paragraphs and their scores.
        """
        self.update_counter()
        if top_k > 0:
            doc_ids, scores = _search_sparse(query, self.index, min(top_k, len(self.paragraphs)))
        else: # then get as many as possible until reaching the max_total_words limit
            doc_ids, scores = _search_sparse(query, self.index, min(100, len(self.paragraphs)))
            total_words = 0
            filtered_doc_ids = []
            filtered_scores = []
            for doc_id, score in zip(doc_ids, scores):
                paragraph = self.paragraphs[doc_id]
                paragraph_words = len(re.split(r'\s+', paragraph.content))
                if total_words + paragraph_words <= max_total_words or len(filtered_scores) == 0:
                    filtered_doc_ids.append(doc_id)
                    filtered_scores.append(score)
                    total_words += paragraph_words
                else:
                    break
            doc_ids, scores = filtered_doc_ids, filtered_scores

        matches = [
            Match(
                search_type="sparse",
                paragraph=self.paragraphs[doc_id],
                preview=None,
                matched_text=None,
                matched_score=score,
                chunk_id=None,
                retriever=self
            )
            for doc_id, score in zip(doc_ids, scores)
        ]
        return matches

    def __str__(self):
        return (
            f"Sparse search (using BM25+) over {os.path.basename(self.index_name)} "
            # f"built from files in folder `{self.folder}` or some relevant files."
        )
    
    def clear(self):
        if hasattr(self, 'index'):
            del self.index
        # remove the index folder to save disk space.
        if os.path.exists(self.index_path):
            os.system(f"rm -r {self.index_path}")


def _build_sparse_index(retriever: SparseRetriever, index_name: str) -> bm25s.BM25:
    """Build BM25 index from a list of paragraphs in a SparseRetriever."""
    paragraphs = retriever.paragraphs
    index_path = retriever.index_path
    if os.path.exists(index_path):
        print_debug(f"Loading existing BM25 index from {index_path}...")
        return bm25s.BM25.load(index_path)
    corpus = [get_paragraph_content(p) for p in paragraphs]
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25(method="bm25+", delta=1.5)
    retriever.index(corpus_tokens, leave_progress=False)
    return retriever


def _search_sparse(query: str, index: bm25s.BM25, top_k: int = -1):
    """
    Sparse (BM25+) search on an BM25+ index.

    Args:
        query: The search query string.
        index: The BM25+ index built from a corpus.
        top_k: The number of top results to return. If -1, return top 50.

    Returns:
        Tuple of (list of document IDs, list of corresponding scores) ordered by score descending.
    """
    # Tokenize query
    query_tokens = bm25s.tokenize(query, stemmer=stemmer)

    # Retrieve
    k = top_k if top_k != -1 else 50
    doc_ids, scores = index.retrieve(query_tokens, k=k)    
    return doc_ids[0], scores[0]


def test_search_sparse():
    # Example usage
    query = "search agent" # "search agent with web search, local index, and local files"
    folder = "references"
    retriever = SparseRetriever(folder=folder, index_name="test_index")
    print(retriever)

    results = retriever.search(query, top_k=5)
    print(f"Search Type: `sparse`")
    for match in results:
        print(f"File: {match.paragraph.filepath}")
        print(f"Score: {match.matched_score}")
        if match.preview:
            print(f"Preview: {match.preview}")
        print("-" * 40)


if __name__ == "__main__":
    test_search_sparse()
