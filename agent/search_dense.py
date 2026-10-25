from .utils import *
from pylate import indexes, models, retrieve
import pickle


class DenseRetriever(Retriever):
    """
    A retriever that performs dense (vector) search using PyLate/ColBERT.

    It builds a PLAID index from `self.paragraphs` on initialization and
    uses a modern (multi-lingual) ColBERT model to encode documents and queries.

    NOTE: PLAID can build a larger index than Voyager, use less GPU by up to 6 times. Using PLAID by default.
    """
    model = models.ColBERT(
        model_name_or_path="lightonai/Reason-ModernColBERT", # 149M param
        # "Qwen/Qwen3-Embedding-0.6B", # seem not as good as Reason-ModernColBERT, in term of agent's final result.
        # "lightonai/GTE-ModernColBERT-v1",
    )
    index_folder = os.path.join(PROJECT_FOLDER, "benchmarks/index/dense")

    def __init__(self, index_name: str, override: bool=False, batch_size: int=4, verbose=False, **kwargs):
        super().__init__(type_="dense", **kwargs)
        self.verbose = verbose
        self.batch_size = batch_size
        self.index_name = index_name
        self.index_path = os.path.join(DenseRetriever.index_folder, index_name)
        self.index = _build_dense_index(self, index_name=index_name, override=override)

    def search(self, query: str, top_k: int = 10, max_total_words: int = 512)-> list[Match]:
        """
        Dense (Late Iteration - ColBERT) search on an ColBERT index.

        Args:
            query: The search query string.
            top_k: The number of top results to return. If -1, return as many as possible until reaching the max_total_words limit.
            max_total_words: The maximum total number of words in the retrieved paragraphs when top_k is -1. Only applicable when top_k is -1.

        Returns:
            A list of `Match` objects containing the retrieved paragraphs and their scores.
        """
        self.update_counter()
        if top_k > 0:
            scores = _search_dense(query, self.index, top_k)
        else: # then get as many as possible until reaching the max_total_words limit
            scores = _search_dense(query, self.index, min(100, len(self.paragraphs)))
            total_words = 0
            filtered_scores = []
            for item in scores:
                doc_id = item["id"]
                paragraph = self.paragraphs[doc_id]
                paragraph_words = len(re.split(r'\s+', paragraph.content))
                if total_words + paragraph_words <= max_total_words or len(filtered_scores) == 0:
                    filtered_scores.append(item)
                    total_words += paragraph_words
                else:
                    break
            scores = filtered_scores

        matches = [
            Match(
                search_type="dense",
                paragraph=self.paragraphs[item["id"]],
                preview=None,
                matched_text=None,
                matched_score=item["score"],
                chunk_id=None,
                retriever=self
            )
            for item in scores
        ]
        return matches

    def __str__(self):
        return (
            f"Dense vector search (using ColBERT) over {os.path.basename(self.index_name)} "
            # f"stored in folder `{self.index_folder}` built from files in folder `{self.folder}`."
        )
    
    def clear(self):
        if hasattr(self, 'index'):
            del self.index
        # remove the index folder to save disk space.
        if os.path.exists(self.index_path):
            os.system(f"rm -r {self.index_path}")


def _build_dense_index(retriever: DenseRetriever, index_name: str, override: bool = False) -> retrieve.ColBERT:
    """Build a ColBERT + Voyager/PLAID index from a list of paragraphs in a DenseRetriever."""
    paragraphs = retriever.paragraphs
    index_exists = os.path.exists(retriever.index_path)
    index = indexes.PLAID(
        index_folder=DenseRetriever.index_folder,
        index_name=index_name,
        override=override,
    )
    if (not index_exists) or override:
        model = DenseRetriever.model
        # load/save cache
        print_debug(f"{retriever.index_path}/documents_embeddings.cache")
        if os.path.exists(f"{retriever.index_path}/documents_embeddings.cache"):
            with open(f"{retriever.index_path}/documents_embeddings.cache", "rb") as f:
                documents_embeddings = pickle.load(f)
        else:
            documents = [get_paragraph_content(p) for p in paragraphs]
            documents_embeddings = model.encode(
                documents,
                batch_size=retriever.batch_size,
                is_query=False,
                show_progress_bar=retriever.verbose,
                # precision="uint8" # it failed due to quantization of non-padding embedding with numpy
            )
            with open(f"{retriever.index_path}/documents_embeddings.cache", "wb") as f:
                pickle.dump(documents_embeddings, f)
        index.add_documents(
            documents_ids=range(len(paragraphs)),
            documents_embeddings=documents_embeddings,
        )
        print_debug(f"Saving ColBERT dense index to {retriever.index_path}...")
    retriever = retrieve.ColBERT(index=index)
    return retriever


def _search_dense(query: str, index: retrieve.ColBERT, top_k: int = -1):
    """
    Dense (Late Iteration - ColBERT) search on an ColBERT index.

    Args:
        query: The search query string.
        index: The ColBERT retriever with a PLAID index built from a corpus.
        top_k: The number of top results to return. If -1, return top 50.

    Returns:
        Tuple of (list of document IDs, list of corresponding scores) ordered by score descending.
    """
    # Encode query
    model = DenseRetriever.model
    queries_embeddings = model.encode([query], batch_size=1, is_query=True)

    # Retrieve
    k = top_k if top_k != -1 else 50
    scores = index.retrieve(queries_embeddings=queries_embeddings, k=k) # num_q x k x {"doc_id", "score"}
    print_debug(scores[0])
    return scores[0]


def test_search_dense():
    query = "search agent"
    folder = "references"
    retriever = DenseRetriever(index_name="test/dense_test_plaid", folder=folder, override=True)
    print(retriever)

    results = retriever.search(query, top_k=10)
    print(f"Search Type: `dense`")
    for match in results:
        if match.paragraph is not None:
            print(f"File: {match.paragraph.filepath}")
        print(f"Score: {match.matched_score}")
        if match.preview:
            print(f"Preview: {match.preview}")
        print("-" * 40)


if __name__ == "__main__":
    test_search_dense()
