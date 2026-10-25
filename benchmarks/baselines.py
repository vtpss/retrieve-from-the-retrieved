from utils import *
from rlm import RLM
# from effgen import load_model
# from effgen.presets import create_agent


### Zeroshot baseline
def zeroshot_call(args, question, given_retrievers, **kwargs):
    return dspy_predict(args, ZeroshotQA)(question=question)

answer_question_description = f"""
You are an expert in question answering and reasoning. Based on your existing knowledge, provide a concise answer with clear reasoning for the following question.
"""

class ZeroshotQA(dspy.Signature):
    # Instruction
    __doc__ = answer_question_description
    # Input
    question: str = dspy.InputField(desc='The question to be answered')
    # Output
    reasoning: str = dspy.OutputField(desc='Your reasoning process for arriving at the answer')
    final_answer: str = dspy.OutputField(desc='The final answer to the question')


### ReAct + Neural Compaction baseline
def effgen_agent(args, question, given_retrievers, **kwargs):
    os.environ["OPENAI_API_KEY"] = args.api_key
    os.environ["OPENAI_BASE_URL"] = args.base_url
    effgen_model = load_model(args.model, provider="openai")
    general_agent = create_agent("general", effgen_model) #, knowledge_base=args.sample_context_folder)   # All tools
    prompt = "Answer the following question with a short answer, perform web search if necessary:\n{}"
    final_answer = general_agent.run(prompt.format(question))
    return {"final_answer": final_answer}


def effgen_demo():
    """
    EffGen seems to be bad. If the result is 0, then it's not meaningful.
    """

    os.environ["OPENAI_API_KEY"] = os.getenv("LITELLM_API_KEY")
    os.environ["OPENAI_BASE_URL"] = os.getenv("LITELLM_BASE_URL")
    # effgen_model = load_model("gpt-oss-120b", provider="openai")
    effgen_model = load_model("kimi-k26-thinking-low", provider="openai")

    # One-line agent creation
    research_agent = create_agent("research", effgen_model) # WebSearch + URLFetch + Wikipedia
    # rag_agent = create_agent("rag", effgen_model, knowledge_base="./docs/")  # RAG pipeline
    general_agent = create_agent("general", effgen_model) #, knowledge_base=args.sample_context_folder)   # All tools
    result = general_agent.run("What is 24344 * 334?")
    print(f"Answer: {result.output}")

    # Prompt
    prompt = "Answer the following question with a short answer, perform web search if necessary:\n{}"

    # SealQA. 
    question = "How many current and former Real Madrid players are ranked in the top 10 of the 2025 Forbes list of the world's highest-paid athletes?"
    result = research_agent.run(prompt.format(question))
    print(f"Answer: {result.output}")
    # Answer: [{'title': "Forbes 2025 World's 10 Highest-Paid Athletes List", 'url': 'https://www.forbes.com/sites/brettknight/2025/05/15/the-worlds-10-highest-paid-athletes-2025/', 'snippet': "May 15, 2025 · On Forbes' list of the top-earning athletes, 10 surpassed $101 million. See who ranked at the top, ... --> effgen returns search results.

    # SimpleQA. EffGen is tailored for SimpleQA.
    question = "How much money, in euros, was the surgeon held responsible for Stella Obasanjo's death ordered to pay her son?"
    result = research_agent.run(prompt.format(question))
    print(f"Answer: {result.output}")
    # Answer: The Spanish surgeon was ordered by the Málaga court to pay **€120 000** in compensation to Stella Obasanjo’s son, Olumuyiwa Obasanjo【5†L1-L4】.
    # Answer: €120,000.


### RLM
def rlm_demo():
    from argparse import Namespace
    args = Namespace(
        base_url="http://0.0.0.0:8000/v1",
        api_key=os.getenv("LITELLM_API_KEY", None),
        model="hosted_vllm/gemma-4-E4B-it",
        debug=True
    )
    init_kwargs = {
        "api_key": os.environ.get("LITELLM_API_KEY", "api_key"),
        "base_url": args.base_url,  # OpenAI backend
        "model_name": args.model[len('hosted_vllm/'):] if args.model.startswith('hosted_vllm/') else args.model
    }
    init_kwargs.update(args.api_kwargs)
    rlm = RLM(
        backend="openai",
        backend_kwargs=init_kwargs,
        verbose=args.debug,
        max_depth=1,    # small LMs can't handle subcall well. Even failed. 
                        # Tried with gpt-oss-20b on officeqa and and failed.
        # max_iterations=60, # default: 30 interations.
    )
    response = rlm.completion("Print me the first 5 powers of two, each on a newline.")
    # response = '1\n2\n4\n8\n16'. Don't have access to metadata like token counts or # iterations if max_depth = 0.
    # response = {'root_model': 'gemma-4-E4B-it', 'prompt': 'Print me the first 5 powers of two, each on a newline.', 'response': 'The content instructs the calculation of the first five powers of two ($2^0$ through $2^4$). The required output, formatted with each number on a new line, is:\n1\n2\n4\n8\n16', 'usage_summary': {'model_usage_summaries': {'gemma-4-E4B-it': {'total_calls': 3, 'total_input_tokens': 6566, 'total_output_tokens': 1183}}}, 'execution_time': 26.23320880997926}


# Prompt for RLM should be straight, to avoid wasting time to determine the task
rlm_answer_question_description = """
You are an expert in question answering and reasoning.

You are doing iterative search and reasoning to find a unique correct answer of a question.
There will be no prior context, but retrievers are provided for you to perform search and retrieval whenever you need. 
Please expect that you may need to perform search and retrieval multiple times, interleaved with your reasoning process, until you are confident about your answer.

Here are instructions to do search/retrieval in your reasoning process:
```python
# **Example**
query = "..."   # To be determined at each search based on the original question and your reasoning process
top_k = 10      # To be determined together with the query. Normally set as 10.
# Call either `web_search`, `dense_search`, or `sparse_search`, based on availability
search_results: list[Match] = web_search(query=query, top_k=top_k)
evidences: list[str] = [paragraph2textattr(result) for result in search_results]    # convert search results to text
```

Available retrievers:
{retrievers}

Please find the answer for the following question:
{question}
"""


def rlm_call(args, question, given_retrievers, **kwargs):
    """
    Make an RLM call. Retries on error with exponential backoff.
    Currently only support OpenAI API backend, i.e., only OpenAI-API-compatible models.
    RLM methods are expected to manage long context automatically.
    """
    init_kwargs = {
        "api_key": args.api_key,
        "base_url": args.base_url,  # OpenAI backend
        "model_name": args.model[len('hosted_vllm/'):] if args.model.startswith('hosted_vllm/') else args.model
    }
    # init_kwargs.update(args.api_kwargs)
    tools = {"Match": Match, "Paragraph": Paragraph, "paragraph2textattr": paragraph2textattr}
    for retriever in given_retrievers:
        if isinstance(retriever, WebRetriever):
            if "rr" in args.agent_workflow:
                def web_search(query, top_k):
                    matched_articles = retriever.search(query=query, top_k=top_k, backend="duckduckgo")
                    matches = rr(args, query, matched_articles)
                    return matches
            else:
                def web_search(query, top_k):
                    return retriever.search(query=query, top_k=top_k, backend="duckduckgo", 
                                            webcontent_format="full_page_markdown")
            tools["web_search"] = web_search
        elif isinstance(retriever, DenseRetriever):
            if "rr" in args.agent_workflow:
                def dense_search(query, top_k):
                    matched_articles = retriever.search(query=query, top_k=top_k)
                    matches = rr(args, query, matched_articles)
                    return matches
            else:
                def dense_search(query, top_k):
                    return retriever.search(query=query, top_k=top_k)
            tools["dense_search"] = dense_search
        elif isinstance(retriever, SparseRetriever):
            if "rr" in args.agent_workflow:
                def sparse_search(query, top_k):
                    matched_articles = retriever.search(query=query, top_k=top_k)
                    matches = rr(args, query, matched_articles)
                    return matches
            else:
                def sparse_search(query, top_k):
                    return retriever.search(query=query, top_k=top_k)
            tools["sparse_search"] = sparse_search

    rlm = RLM(
        backend="openai",
        backend_kwargs=init_kwargs,
        verbose=args.debug,
        max_depth=1,    # small LMs can't handle subcall well. Even failed. 
                        # Tried with gpt-oss-20b on officeqa and and failed.
        max_iterations=args.max_iterations+10, # default: 30 interations.
        compaction=True,
        custom_tools=tools,
        custom_sub_tools=tools,
        persistent=True
    )

    for _ in range(1, args.max_retries+1):
        try:
            prompt = rlm_answer_question_description.format(
                question=question, retrievers=str_enumerate(given_retrievers))
            # question = f"Perform necessary searches and answer the following question '{question}'"
            response = rlm.completion(
                root_prompt=prompt,     # suggested by RLM docs to pass the main question here
                prompt="",          # context for QA, as orginally design
            )
            if isinstance(response, str):   # in case max depth == 0
                response = {"response": response}
            else:
                response = response.to_dict()

            # extract text inside FINAL(<text>) if present
            match = re.search(r"FINAL\((.*?)\)", response.get("response", ""), flags=re.DOTALL)
            response["final_answer"] = match.group(1).strip() if match else response.get("response", "").strip()
            return response

        except Exception as e:
            print_debug(f"LLM call error: {e}")
            print_debug(f"LLM call re-trying {_}-th time ...")
            time.sleep(2)
