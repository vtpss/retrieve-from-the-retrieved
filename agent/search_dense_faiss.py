"""FAISS dense searcher for loading an existing index and querying it."""

import os
import glob
import logging
import pickle
from typing import Any, Dict, List, Optional

import torch
import faiss
import numpy as np
from transformers import AutoTokenizer
from tevatron.retriever.arguments import ModelArguments
from tevatron.retriever.driver.encode import DenseModel
from tevatron.retriever.searcher import FaissFlatSearcher
from .utils import Retriever, Paragraph, Match, print_debug

logger = logging.getLogger(__name__)


class FaissRetriever(Retriever):
    model_name = "Qwen/Qwen3-Embedding-8B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pooling = "eos"
    normalize = False
    task_prefix = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"

    def __init__(self, index_path: str, docid_to_paragraph: dict, **kwargs):
        super().__init__(type_="dense", **kwargs)
        if not index_path:
            raise ValueError("index_path is required for FAISS searcher")

        self.docid_to_paragraph = docid_to_paragraph
        self.index_path = os.path.join(index_path, self.model_name.rsplit('/', 1)[-1].lower(), "*")
        self.index_name = ""
        self.retriever = None
        self.model = None
        self.tokenizer = None
        self.lookup = []
        logger.info("Initializing FAISS searcher...")

        self._load_index()
        self._load_model()
        logger.info("FAISS searcher initialized successfully")

    def _load_index(self) -> None:
        def pickle_load(path):
            with open(path, "rb") as f:
                reps, lookup = pickle.load(f)
            return np.asarray(reps), lookup

        index_files = sorted(glob.glob(self.index_path))
        logger.info("Pattern match found %d files; loading index shards.", len(index_files))
        if not index_files:
            raise ValueError(f"No files found matching pattern: {self.index_path}")

        first_reps, first_lookup = pickle_load(index_files[0])
        self.retriever = FaissFlatSearcher(first_reps)
        self.lookup = list(first_lookup)
        for index_file in index_files[1:]:
            shard_reps, shard_lookup = pickle_load(index_file)
            self.retriever.add(shard_reps)
            self.lookup.extend(shard_lookup)

        self._setup_gpu()

    def _setup_gpu(self) -> None:
        num_gpus = faiss.get_num_gpus()
        logger.info("Using %d GPU(s)", num_gpus)

        if num_gpus == 0:
            logger.info("No GPU found or using faiss-cpu. Using CPU.")
        elif num_gpus == 1:
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            res = faiss.StandardGpuResources()
            self.retriever.index = faiss.index_cpu_to_gpu(res, 0, self.retriever.index, co)
        else:
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = True
            self.retriever.index = faiss.index_cpu_to_all_gpus(self.retriever.index, co, ngpu=num_gpus)

    def _load_model(self) -> None:
        logger.info("Loading model: %s", self.model_name)
        cache_dir = os.getenv("HF_HOME") or None
        model_args = ModelArguments(
            model_name_or_path=self.model_name,
            normalize=self.normalize,
            pooling=self.pooling,
            cache_dir=cache_dir,
        )

        self.model = DenseModel.load(
            model_args.model_name_or_path,
            pooling=model_args.pooling,
            normalize=model_args.normalize,
            lora_name_or_path=model_args.lora_name_or_path,
            cache_dir=model_args.cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            attn_implementation=model_args.attn_implementation,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            padding_side="left",
        )
        logger.info("Model loaded successfully")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not all([self.retriever, self.model, self.tokenizer, self.lookup]):
            raise RuntimeError("Searcher not properly initialized")

        batch_dict = self.tokenizer(
            self.task_prefix + query,
            padding=True,
            truncation=True,
            max_length=1024*8,
            return_tensors="pt",
        )
        batch_dict = {key: value.to(self.device) for key, value in batch_dict.items()}
        with torch.no_grad():
            q_reps = self.model.encode_query(batch_dict)
            q_reps = q_reps.cpu().detach().numpy()
        all_scores, psg_indices = self.retriever.search(q_reps, top_k)

        results = []
        for score, index in zip(all_scores[0], psg_indices[0]):
            results.append({"id": self.lookup[index], "score": float(score)})
        print_debug("Original FAISS Index:", results)
        matches = [
            Match(
                search_type="dense",
                paragraph=self.docid_to_paragraph[item["id"]],
                preview=None,
                matched_text=None,
                matched_score=item["score"],
                chunk_id=None,
                retriever=self
            )
            for item in results
        ]
        return matches

    def __str__(self):
        return f"Dense vector search (using Faiss) over {os.path.basename(self.index_name)}"
