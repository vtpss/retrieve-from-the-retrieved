from .dspy_programs import *
from .search_dense import DenseRetriever
from .search_sparse import SparseRetriever
from .search_web import NEWS_AGENCY_HOSTNAME_PROPERTY, WebRetriever, normalize_str
from .utils import *


# Helper functions
def paragraph2textattr(p: Union[Paragraph, Match]) -> str:
    if isinstance(p, Match):
        p = p.paragraph
    return paragraph_to_text(p, agency_attribute_map=NEWS_AGENCY_HOSTNAME_PROPERTY)


def evidences_to_text(evidences: list[Paragraph]):
    sep = "\n\n---\n\n"
    return sep.join(
        [f"Evidence {i}:\n{paragraph2textattr(e)}\n" for i, e in enumerate(evidences)]
    )


# Later, can turn this function into an Object, to make the flow clear.
def search_and_reason(
    args,
    question: str,
    given_retrievers: list[Retriever],
    sample_id: str = None,
) -> str:
    """
    A search function in the ReAct style that iteratively
    1. THOUGHT: form a query based on the current search and reasoning state
    2. ACTION: perform search on web, local index, or file search based on the query.
        - also manage the search output and overall context/memory
    3. OBSERVATION: attempt to answer the question based on the retrieved information
        and return candidates and status (finished, lacked of info, ..)
    4. REPEAT: repeat until a final answer is produced or max iterations is reached.

    Args:
        args: The arguments containing model and other configurations for LLM calls.
        question: The question to be answered.
        given_retrievers: A list of retriever objects to perform search.

    Returns:
        response: A dictionary containing the final_answer, reasoning, candidates, and other data fields.
    """
    all_retrievers = given_retrievers.copy()  # just need shallow copy
    all_queries = []
    all_hrefs = []
    all_files = []
    all_evidences: list[Paragraph] = []
    all_core_infos: list[str] = []
    all_statuses = ["START"]
    candidates = []
    start_time = time.time()
    llm_input_words = []  # len(text.split()). Use this simple approximation.
    dense_corpus_words = []

    safemodel = args.model.split("/")[-1]
    save_path = f"results/llm_calls/{safemodel}/{args.benchmark}_{sample_id}_{len(question)}.jsonl"
    fout = (
        open(save_path, "w", encoding="utf-8")
        if args.save_llm_calls and sample_id
        else None
    )

    for iteration in range(args.max_iterations):
        retriever_metadata = [
            f"{i}. {retriever}" for i, retriever in enumerate(all_retrievers)
        ]
        evidences_text = evidences_to_text(all_evidences)
        print_debug(f"# {'=' * 5} Iteration {iteration + 1} {'=' * 5}")
        print_debug(f"\n**Question**: \n\n'{question}'")
        print_debug(f"\n**Candidate Answers**: \n\n{str_enumerate(candidates)}")
        print_debug(f"\n**All Statuses**: \n\n{str_enumerate(all_statuses)}")
        print_debug(f"\n**All Queries**: \n\n{str_enumerate(all_queries)}")
        print_debug(f"\n**All Evidences**: \n\n{evidences_text}")
        print_debug(f"\n**Available Retrievers**: \n\n{str_enumerate(all_retrievers)}")
        print_hyphen(10)

        # 1. THOUGHT: Form a query based on the current search and reasoning state
        llm_input_words.append(
            next_search_query_description_wordcount
            + len(
                (
                    question
                    + str_enumerate(candidates)
                    + str_enumerate(all_statuses)
                    + "\n".join(all_queries)
                    + evidences_text
                    + "\n".join(retriever_metadata)
                ).split()
            )
        )
        next_query, retriever_ids = dspy_predict(args, NextSearchQuery, fout=fout)(
            question=question,
            candidate_answers=str_enumerate(candidates),
            search_reasoning_history=str_enumerate(all_statuses),
            searched_queries="\n".join(all_queries),
            all_evidences=evidences_text,
            available_retrievers="\n".join(retriever_metadata),
        ).values()
        next_query = re.sub(
            r'["“”]', "", next_query
        )  # remove quotes for better search results.
        all_queries.append(next_query)
        print_debug(f"**Next Query**:\n\n{next_query}\n")
        print_debug(
            f"**Retrievers**:\n\n{'\n'.join([retriever_metadata[i] for i in retriever_ids])}\n"
        )
        print_hyphen(10)

        # 2. ACTION: Perform search based on the new query and manage the search result
        retrieved_evidences = []
        for id_ in retriever_ids:
            retriever = all_retrievers[id_]

            if retriever in given_retrievers:
                print_debug(f"## Performing search on a given retriever {retriever}\n")
                kwargs = {}
                if isinstance(retriever, WebRetriever) and args.chunking == "fix-length":
                    kwargs["webcontent_format"] = "full_page_markdown"
                matched_articles = retriever.search(next_query, args.top_k, **kwargs)
                all_hrefs.extend(
                    [a.paragraph.metadata.get("url", "") for a in matched_articles]
                )
                all_files.extend([a.paragraph.filepath for a in matched_articles])

                print_debug()
                paragraphs = [m.paragraph for m in matched_articles]
                remained_paragraphs = []
                for p in paragraphs:
                    if isinstance(p, Table):
                        remained_paragraphs.append(p)
                    else:
                        remained_paragraphs.extend(chunk_paragraph(p, paragraph_sep="NO_PARAGRAPH_SEP" if args.chunking == "fix-length" else "\n\n" ))
                print_debug(f"### Compacting evidences for {retriever} ...\n")

                if "sparse" in args.context_filtering_with:
                    # 2.1. clean the context with cheap sparse search
                    s_retriever = SparseRetriever(
                        index_name=normalize_str(next_query), paragraphs=remained_paragraphs
                    )
                    matches = s_retriever.search(next_query, top_k=args.sparse_top_k)
                    total_words = sum([len(m.paragraph.content.split()) for m in matches])
                    print_debug(
                        f"\n\nExtracting {len(matches)} evidences of {total_words} words with {s_retriever}"
                    )
                    print_debug("```python")
                    for s in matches:
                        print_debug(s)
                    print_debug("```\n")
                    # all_retrievers.append(s_retriever)
                    s_retriever.clear()
                    del s_retriever

                    # 2.2. span paragraph to avoid missing context
                    remained_paragraphs = [m.paragraph for m in matches]
                    remained_paragraphs = span_paragraphs(
                        remained_paragraphs, window_size=args.paragraph_chain_window
                    )
                    print_debug(
                        f"### After spanning paragraphs with window size of {args.paragraph_chain_window}, we have {len(remained_paragraphs)} paragraphs\n"
                    )
                    print_debug("```python")
                    for p in remained_paragraphs:
                        print_debug(p)
                    print_debug("```\n")
                    print_hyphen(10)

                if "llm" in args.context_filtering_with:
                    # summarize all evidence into a single piece of text using LLM
                    print_debug(
                        f"### Summarizing {len(remained_paragraphs)} evidences into a single piece of text using LLM ...\n"
                    )
                    llm_input_words.append([
                        compact_evidence_description_wordcount
                        + len((question + evidences_to_text(remained_paragraphs)).split())
                    ])
                    compacted_evidence = dspy_predict(args, CompactEvidences, is_compaction_call=True)(
                        question=question,
                        all_evidences=evidences_to_text(remained_paragraphs)
                    )["compacted_evidence"].strip()
                    remained_paragraphs = [Paragraph(content=compacted_evidence, filepath="llm_compacted_evidence", metadata={})]

                elif "dense" in args.context_filtering_with:
                    # 2.3. build dense retriever on the filtered paragraphs, and filter once more
                    dense_corpus_words.append(
                        sum([len(p.content.split()) for p in remained_paragraphs])
                    )
                    print_debug(
                        f"### Building dense retriever based on the compacted evidences ...\n"
                    )
                    d_retriever = DenseRetriever(
                        index_name=f"{args.benchmark}/web-search-clean-result-{normalize_str(next_query)}",
                        paragraphs=remained_paragraphs,
                        override=True,
                    )
                    matches = d_retriever.search(next_query, top_k=args.dense_top_k)
                    print_debug()
                    print_debug(
                        f"### Compacted evidences from dense retriever {d_retriever}:\n"
                    )
                    total_words = sum([len(m.paragraph.content.split()) for m in matches])
                    print_debug(
                        f"Extracting top {len(matches)} evidences of {total_words} words ...\n"
                    )
                    print_debug("```python")
                    for m in matches:
                        print_debug(m)
                    print_debug("```\n")
                    print_hyphen(10)
                    # all_retrievers.append(d_retriever)
                    d_retriever.clear()
                    del d_retriever
                    remained_paragraphs = [m.paragraph for m in matches]

                # 2.4. finally, add the dense search result into the retrieved evidences
                retrieved_evidences.extend(remained_paragraphs)

            else:
                print_debug(f"## {retriever_metadata[id_]}:\n")
                matches = retriever.search(next_query, args.top_k)

                print_debug("```python")
                for m in matches:
                    print_debug(m)
                print_debug("```\n")
                print_hyphen(10)
                retrieved_evidences.extend([m.paragraph for m in matches])

        # compact and rewrite new retrieved evidences to keep only relevant parts
        """
        total_words = sum([len(e.content.split()) for e in retrieved_evidences])
        print_debug(f"## Compacting and rewriting {len(retrieved_evidences)} new evidences of {total_words} words ...")
        compaction_input_words = []
        for e in tqdm(retrieved_evidences, desc="Compacting evidences", total=len(retrieved_evidences)):
            if isinstance(e, Table): # don't compact table, keep it as is.
                continue
            compaction_input_words.append(compact_evidence_description_wordcount + len((question + e.content).split()))
            compacted_evidence = dspy_predict(args, CompactEvidence, is_compaction_call=True, fout=fout)(
                question=question,
                evidence=e.content
            )["compacted_evidence"].strip()
            if "INTACT" in compacted_evidence:
                pass
            elif compacted_evidence:
                e.original = e.content[:400] + ' ...'
                e.content = compacted_evidence
                e.preview = compacted_evidence[:200]
            else:
                retrieved_evidences.remove(e)
        llm_input_words.append(compaction_input_words)

        total_words = sum([len(e.content.split()) for e in retrieved_evidences])
        print_debug(f"\nAfter compaction: {len(retrieved_evidences)} evidences remained with {total_words} words.")
        print_debug('```python')
        for e in retrieved_evidences:
            print_debug(e)
        print_debug('```\n')
        print_hyphen(10)
        """

        # add new evidences
        count_new_evidences = 0
        for e in retrieved_evidences:
            if e not in all_evidences:
                all_evidences.append(e)
                count_new_evidences += 1
        total_words = sum([len(e.content.split()) for e in all_evidences])
        print_debug(f"Retrieved {count_new_evidences} new evidences.")
        print_debug(
            f"Having {len(all_evidences)} evidences of {total_words} words in total."
        )
        print_hyphen(10)

        # 3. OBSERVATION: Attempt to answer the question based on the retrieved information
        print_debug(
            "## Attempting to answer the question based on retrieved evidences ...\n"
        )
        evidences_text = evidences_to_text(all_evidences)
        llm_input_words.append(
            answer_question_description_wordcount
            + len((question + str_enumerate(candidates) + evidences_text).split())
        )
        Signature = (
            AnswerQuestion
            if iteration < args.max_iterations - 1
            else AnswerQuestionFinalAttempt
        )
        response = dspy_predict(args, Signature, fout=fout)(
            question=question,
            previous_candidate_answers=str_enumerate(candidates),
            all_evidences=evidences_text,
        )
        reasoning, candidates, status, confidence, not_useful_evidence_ids = (
            response.values()
        )
        print_debug(f"**Reasoning**: \n\n{reasoning}\n")
        print_debug(f"**Answer Candidates**:\n\n{str_enumerate(candidates)}\n")
        print_debug(f"**Status**: \n\n{status}\n")
        print_debug(f"**Confidence**: \n\n{confidence}\n")
        print_debug(f"**Not Useful Evidence IDs**: \n\n{not_useful_evidence_ids}\n")

        # manage the overall context/memory and search status
        all_statuses.append(status)
        if (
            status == "FINISH" and (len(candidates) == 1)
        ) or iteration == args.max_iterations - 1:
            response = {
                "final_answer": candidates[0] if candidates else "No answer",
                "reasoning": reasoning,
                "candidates": candidates,
                "confidence": confidence,
                "all_evidences": [
                    {"filepath": e.filepath, "content": e.content}
                    for e in all_evidences
                ],
                # "all_retrievers": all_retrievers,
                "all_queries": all_queries,
                "all_hrefs": sorted(set(all_hrefs)),
                "all_files": [os.path.basename(f) for f in sorted(set(all_files))],
                "all_statuses": all_statuses,
                "run_time": time.time() - start_time,
                "num_iterations": iteration + 1,
                "llm_input_words": llm_input_words,
                "dense_corpus_words": dense_corpus_words,
            }
            break
        else:
            all_evidences = [
                e for i, e in enumerate(all_evidences) if i not in not_useful_evidence_ids
            ]
            print_debug(
                f"After removing not useful evidences, we have {len(all_evidences)} evidences in total.\n\n"
            )

    print_debug(
        f"\n\n### Search results for `{question}` is: `{candidates[0] if candidates else 'No answer'}`\n"
    )
    return response


def rr(args, query, matched_articles):
    paragraphs = [m.paragraph for m in matched_articles]
    remained_paragraphs = []
    for p in paragraphs:
        if isinstance(p, Table):
            remained_paragraphs.append(p)
        else:
            remained_paragraphs.extend(chunk_paragraph(p, paragraph_sep="NO_PARAGRAPH_SEP" if args.chunking == "fix-length" else "\n\n"))
    
    s_retriever = SparseRetriever(
        index_name=normalize_str(query), paragraphs=remained_paragraphs
    )
    matches = s_retriever.search(query, top_k=args.sparse_top_k)
    s_retriever.clear()
    del s_retriever
    remained_paragraphs = [m.paragraph for m in matches]
    remained_paragraphs = span_paragraphs(
        remained_paragraphs, window_size=args.paragraph_chain_window
    )

    d_retriever = DenseRetriever(
        index_name=f"{args.benchmark}/web-search-clean-result-{normalize_str(query)}",
        paragraphs=remained_paragraphs,
        override=True,
    )
    matches = d_retriever.search(query, top_k=args.dense_top_k)
    d_retriever.clear()
    del d_retriever
    # remained_paragraphs = [m.paragraph for m in matches]
    # [paragraph2textattr(p) for p in remained_paragraphs]
    return matches

def add_retrieval_args(args):
    args.top_k = 10
    base_retrieval_effort = dict(
        dense_top_k=5,  # for dense search, return more results to increase coverage
        sparse_top_k=10,  # for sparse search, return more results to increase coverage
        paragraph_chain_window=1,  # neighbor paragraphs within the window that may contain useful information
    )
    retrieval_effort_multiplier_mapping = {
        "minimal": [1, 1, 0],
        "low": [2, 3, 1],
        "medium": [3, 4, 1],
        "high": [4, 5, 1],
        "medium-no-window": [3, 4, 0],
    }
    for key, multiplier in zip(
        base_retrieval_effort.keys(),
        retrieval_effort_multiplier_mapping[args.retrieval_effort],
    ):
        setattr(args, key, base_retrieval_effort[key] * multiplier)

    if args.context_filtering_with == "all":
        args.context_filtering_with = "sparse,dense"
    # elif args.context_filtering_with == "sparse":
    #     args.sparse_top_k = args.dense_top_k

def check_reference_hit(gt_urls, pred_urls):
    if isinstance(gt_urls, str):
        gt_urls = [gt_urls]
    if isinstance(pred_urls, str):
        pred_urls = [pred_urls]
    hit_urls = []
    for url in gt_urls:
        url_ = url.split("#", 1)[
            0
        ]  # ignore the part after #, as it usually doesn't affect the content.
        if any([url_ in p for p in pred_urls]):
            hit_urls.append(url)
    return hit_urls


def demo(i, search_agent=search_and_reason):
    from argparse import Namespace

    args = Namespace(
        base_url=os.getenv("LITELLM_BASE_URL", "http://0.0.0.0:8000/v1"),
        api_key=os.getenv("LITELLM_API_KEY", None),
        model=os.getenv("LITELLM_MODEL", "hosted_vllm/gemma-4-E4B-it"),
        llm_call_timeout=100,
        # reasoning_effort="medium",
        llm_call_sleep=0,
        retrieval_effort=os.getenv("RETRIEVAL_EFFORT", "medium"),  # minimal, low, medium, high
        benchmark="test",
        max_iterations=5,
        context_filtering_with="all",
        save_llm_calls=False,
        use_separated_judge=False,
        use_separated_summarizer=False,
        chunking="document-structure",
        agent_workflow="",
    )
    add_retrieval_args(args)
    retrievers = [
        WebRetriever(),
        # DenseRetriever(),
        # SparseRetriever(),
    ]

    samples = [
        dict(
            question=f"How many current and former Real Madrid players are ranked in the top 10 of the 2025 Forbes list of the world's highest-paid athletes?",
            ref_url=[
                "https://en.wikipedia.org/wiki/Forbes_list_of_the_world%27s_highest-paid_athletes#2025_list"
            ],
            gt_answer="2",  # as of April 2026.
        ),
        dict(
            question="How many YouTube music videos have surpassed 7 billion views?",
            ref_url=[
                "https://en.wikipedia.org/wiki/List_of_most-viewed_YouTube_videos#:~:text=Baby%20Shark%20Dance%5B,October%208%2C%202016"
            ],
            gt_answer="5",  # as of April 2026.
        ),
        dict(  # truely advanced table reasoning
            question="What is the most recent film to join the top 10 highest-grossing films of all time?",
            url=[
                "https://en.wikipedia.org/wiki/List_of_highest-grossing_films#Highest-grossing_films:~:text=the%20world.-,Highest%2Dgrossing%20films,-%5B12%5D"
            ],
            gt_answer="Zootopia 2",
        ),
        dict(  # "experimental physicist" in this question is a good identifier. John M. Martinis could be the person, and 68 could be the correct answer.
            question="What is the current age of the last American experimental physicist to win the Nobel Prize in Physics?",
            url=[
                "https://en.wikipedia.org/wiki/List_of_Nobel_laureates_in_Physics#:~:text=John%20Clauser%0A(b,American\r",
                "https://en.wikipedia.org/wiki/John_Clarke_(physicist)",
            ],
            gt_answer="84 years old",
        ),
        dict(  # Google / Wikipedia search seems to be better than DuckDuckGo. Probably need to increase the search effort to 10 as well.
            question="Which organization is the world's largest card payment processing organization, providing credit cards, debit cards, and electronic payment systems?",
            url=[
                "https://en.wikipedia.org/wiki/UnionPay#:~:text=In%202015%2C%20the%20UnionPay%20overtook%20Visa%20and%20Mastercard%20in%20total%20value%20of%20payments%20made%20by%20customers%20and%20became%20the%20largest%20card%20payment%20processing%20organization%20(debit%20and%20credit%20cards%20combined)%20in%20the%20world%20surpassing%20the%20two",
                "",
            ],
            gt_answer="UnionPay",
        ),
    ]
    date = "2026-04-30"
    question = samples[i]["question"]
    start_time = time.time()
    print_debug(f"\nConfigurations\n```python\n{args}\n```\n")
    response = search_agent(
        args=args, question=question, given_retrievers=retrievers
    )
    print("\n" + "=" * 20)
    print(f"Question: {question}")
    print(f"Correct answer: {samples[i]['gt_answer']}")
    print(f"Predicted answer: {response['final_answer']}")
    print(f"Search iterations: {response['num_iterations']}")
    print(f"Total input tokens to LLMs: {sum(response['llm_input_words'])}")
    print(f"Total input tokens to dense-search embedding model: {sum(response['dense_corpus_words'])}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    os.environ["DEBUG"] = "1"
    os.environ["RETRIEVAL_EFFORT"] = "medium"
    demo(4)
