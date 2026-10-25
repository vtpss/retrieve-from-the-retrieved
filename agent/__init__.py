# from pathlib import Path
# import sys
# sys.path.append(str(Path(__file__).parent))  # Add current directory to path for imports

from .utils import *
from .search_sparse import SparseRetriever
from .search_dense import DenseRetriever
from .search_dense_faiss import FaissRetriever
from .search_web import WebRetriever, METADATA_TEXT_SEP, normalize_str
from .dspy_programs import *
from .main import (
    search_and_reason as pss_agent_call, rr, check_reference_hit,
    add_retrieval_args, paragraph2textattr, str_enumerate
)
from .react import react_call

__version__ = '0.2.2' # In this version, by default, we only use duckduckgo search, and don't use code.
