from .utils import *
from .dspy_programs import *
from .search_sparse import SparseRetriever
from .main import evidences_to_text, demo


def react_call(
    args,
    question: str,
    given_retrievers: list[Retriever],
    **kwargs
) -> str:
    """
    A search function in the ReAct style that iteratively
    1. THOUGHT: form a query based on the current search and reasoning state
    2. ACTION: perform search on web, local index, or file search based on the query.
        - also compact the context (i.e., retrieved evidences) if it reaches the default threshold 0.66. 
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
    all_retrievers = given_retrievers.copy() # just need shallow copy
    all_queries = []
    all_hrefs = []
    all_files = []
    all_evidences: list[Paragraph] = []
    all_core_infos: list[str] = []
    all_statuses = ["START"]
    candidates = []
    start_time = time.time()
    llm_input_words = []    # len(text.split()). Use this simple approximation.
    dense_corpus_words = []

    for iteration in range(args.max_iterations):
        retriever_metadata = [f"{i}. {retriever}" for i, retriever in enumerate(all_retrievers)]
        evidences_text = evidences_to_text(all_evidences)
        print_debug(f"# {'='*5} Iteration {iteration+1} {'='*5}")
        print_debug(f"\n**Question**: \n\n'{question}'")
        print_debug(f"\n**Candidate Answers**: \n\n{str_enumerate(candidates)}")
        print_debug(f"\n**All Statuses**: \n\n{str_enumerate(all_statuses)}")
        print_debug(f"\n**All Queries**: \n\n{str_enumerate(all_queries)}")
        print_debug(f"\n**All Evidences**: \n\n{evidences_text}")
        print_debug(f"\n**Available Retrievers**: \n\n{str_enumerate(all_retrievers)}")
        print_hyphen(10)

        # 1. THOUGHT: Form a query based on the current search and reasoning state
        llm_input_words.append(next_search_query_description_wordcount + len((
            question + str_enumerate(candidates) + str_enumerate(all_statuses) + \
            "\n".join(all_queries) + evidences_text + "\n".join(retriever_metadata)
        ).split()))
        if args.max_iterations <= 1:
            next_query, retriever_ids = question, list(range(len(given_retrievers)))
        else:
            next_query, retriever_ids = dspy_predict(args, NextSearchQuery)(
                question=question,
                candidate_answers=str_enumerate(candidates),
                search_reasoning_history=str_enumerate(all_statuses),
                searched_queries="\n".join(all_queries),
                all_evidences=evidences_text,
                available_retrievers="\n".join(retriever_metadata),
            ).values()
        next_query = re.sub(r'["“”]', "", next_query) # remove quotes for better search results.
        all_queries.append(next_query)
        print_debug(f"**Next Query**:\n\n{next_query}\n")
        print_debug(f"**Retrievers**:\n\n{"\n".join([retriever_metadata[i] for i in retriever_ids])}\n")
        print_hyphen(10)

        # 2. ACTION: Perform search based on the new query and manage the search result
        retrieved_evidences = []
        for id_ in retriever_ids:
            retriever = all_retrievers[id_]
            print_debug(f"## Performing search on a given retriever {retriever}\n")
            kwargs = {"webcontent_format": "full_page_markdown"} if "web" in str(retriever).lower() else {}
            matched_articles = retriever.search(next_query, args.top_k, **kwargs)
            all_hrefs.extend([a.paragraph.metadata.get("url", "") for a in matched_articles])
            all_files.extend([a.paragraph.filepath for a in matched_articles])
            retrieved_evidences.extend([m.paragraph for m in matched_articles])

            if "slim" in args.agent_workflow:
                print_debug(f"### Retrieved {len(matched_articles)} evidences from {retriever}.\n")
                for i in range(len(retrieved_evidences)):
                    evid = retrieved_evidences[i]
                    chunks = chunk_long_paragraph_by_max_chars(evid.content, max_char=1024*3, regex_clean_cut="\n\n")
                    paragraphs = [Paragraph(content=chunk, filepath=evid.filepath, metadata=evid.metadata) for chunk in chunks]
                    s_retriever = SparseRetriever(
                        index_name=next_query, paragraphs=paragraphs
                    )
                    matches = s_retriever.search(next_query, top_k=1)
                    retrieved_evidences[i] = matches[0].paragraph if matches else evid

        # 2.1. Neural Compaction: Rewrite new retrieved evidences to keep only relevant parts
        retrieved_evidences = list(set(retrieved_evidences)) # remove duplicates
        total_words = sum([len(e.content.split()) for e in retrieved_evidences])
        total_tokens = sum([count_tokens(e.content) for e in retrieved_evidences])

        if args.max_iterations <= 1:
            for e in retrieved_evidences:
                e.content = " ".join(e.content.split()[:1024*4])

        elif total_tokens > int(1024*128*0.5): # 0.66 is the default threshold for compaction
            print_debug(f"## Compacting {len(retrieved_evidences)} evidences of {total_words} words ...")
            compaction_input_words = []
            for e in tqdm(retrieved_evidences, desc="Compacting evidences", total=len(retrieved_evidences)):
                # if isinstance(e, Table): # don't compact table, keep it as is.
                #     continue
                try:
                    compacted_evidence = dspy_predict(args, CompactEvidence, is_compaction_call=True)(
                        question=question,
                        evidence=e.content[:1024*6*16] # 64k tokens ~ to chars
                    )["compacted_evidence"].strip()
                    compaction_input_words.append(compact_evidence_description_wordcount + len((question + e.content).split()))
                except:
                    compacted_evidence = ""

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

        # add new evidences
        count_new_evidences = 0
        for e in retrieved_evidences:
            if e not in all_evidences:
                all_evidences.append(e)
                count_new_evidences += 1

        if count_tokens(evidences_to_text(all_evidences)) > int(1024*128*0.8):
            llm_input_words.append([
                compact_evidence_description_wordcount
                + len((question + evidences_to_text(all_evidences)).split())
            ])
            compacted_evidence = dspy_predict(args, CompactEvidences, is_compaction_call=True)(
                question=question,
                all_evidences=evidences_to_text(all_evidences)
            )["compacted_evidence"].strip()
            all_evidences = [Paragraph(content=compacted_evidence, filepath="llm_compacted_evidence", metadata={})]

        total_words = sum([len(e.content.split()) for e in all_evidences])
        print_debug(f"Retrieved {count_new_evidences} new evidences.")
        print_debug(f"Having {len(all_evidences)} evidences of {total_words} words in total.")
        print_hyphen(10)

        # 3. OBSERVATION: Attempt to answer the question based on the retrieved information
        print_debug("## Attempting to answer the question based on retrieved evidences ...\n")
        evidences_text = evidences_to_text(all_evidences)
        llm_input_words.append(answer_question_description_wordcount + len((
            question + str_enumerate(candidates) + evidences_text
        ).split()))
        Signature = AnswerQuestion if iteration < args.max_iterations - 1 else AnswerQuestionFinalAttempt
        response = dspy_predict(args, Signature)(
            question=question,
            previous_candidate_answers=str_enumerate(candidates),
            all_evidences=evidences_text
        )
        reasoning, candidates, status, confidence, not_useful_evidence_ids = response.values()
        print_debug(f"**Reasoning**: \n\n{reasoning}\n")
        print_debug(f"**Answer Candidates**:\n\n{str_enumerate(candidates)}\n")
        print_debug(f"**Status**: \n\n{status}\n")
        print_debug(f"**Confidence**: \n\n{confidence}\n")
        print_debug(f"**Not Useful Evidence IDs**: \n\n{not_useful_evidence_ids}\n")

        # manage the overall context/memory and search status
        all_statuses.append(status)
        if (status == "FINISH" and (len(candidates) == 1)) or iteration == args.max_iterations - 1:
            response = {
                "final_answer": candidates[0] if candidates else 'No answer',
                "reasoning": reasoning,
                "candidates": candidates,
                "confidence": confidence,
                "all_evidences": [{'filepath': e.filepath, 'content': e.content} for e in all_evidences],
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
            all_evidences = [e for i, e in enumerate(all_evidences) if i not in not_useful_evidence_ids]
            print_debug(f"After removing not useful evidences, we have {len(all_evidences)} evidences in total.\n\n")

    print_debug(f"\n\n### Search results for `{question}` is: `{candidates[0] if candidates else 'No answer'}`\n")
    return response


if __name__ == "__main__":
    os.environ["DEBUG"] = "1"
    os.environ["RETRIEVAL_EFFORT"] = "medium"
    demo(4, react_call)
