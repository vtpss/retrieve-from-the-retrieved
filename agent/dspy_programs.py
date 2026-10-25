import dspy
import litellm
from .utils import *
import time, datetime
current_month = datetime.date.today().strftime("%b, %Y")

# --------------------
# Prompt and LLM Calls Setup
# --------------------

# https://dspy.ai/tutorials/cache/#disablingenabling-dspy-cache
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)

# convert DSPy program to generic prompt. the magic is here
# https://github.com/stanfordnlp/dspy/blob/main/dspy/adapters/chat_adapter.py#L34
DSPyChatAdapter = dspy.ChatAdapter()
def get_messages_templates(dspy_signature, demos: list[dspy.Example]=[]):
    generic_messages = DSPyChatAdapter.format(
        dspy_signature,
        demos=demos,
        inputs={k: f"{{{k}}}" for k in dspy_signature.input_fields},
    )
    return generic_messages

# full pipeline
def dspy_simple_format_correction(text):
    for incorr, corr in [('[[##', '[[ ##'), ('##]]', '## ]]'), ('[[# ', '[[ ##'), (' #]]', '## ]]')]:
        text = text.replace(incorr, corr)
    text = text.replace('[[', '\n[[').replace(']]', ']]\n')
    return text

def dspy_predict(args, signature: dspy.Signature, demos=[], **kwargs):
    parser = lambda x: DSPyChatAdapter.parse(signature, x)
    messages = get_messages_templates(signature, demos=demos)
    def _predict(**inputs):
        messages[-1]["content"] = messages[-1]["content"].format(**inputs)
        try:
            parsed_output = llm_call(args, messages, parser=parser, **kwargs)
            return parsed_output
        except Exception as e:
            print_debug("Error parsing LLM output:", e)
            return None
    return _predict

def llm_call(args, messages, get_thinking_tokens=False, parser=lambda x:x,
             is_llm_judge_call=False, is_compaction_call=False, fout=None, **kwargs):
    """
    Make an LLM call with the given messages and arguments. Retries on error after 1 sec.
    NOTE: vLLM deploys models with default model config,
        e.g., the model's `generation_config.json`: `{'temperature': 1.0, 'top_k': 64, 'top_p': 0.95}`
    """
    kwargs = {}
    # if "qwen3." in args.model.lower(): # disable thinking for Qwen 3.5/6, as it thinks too much.
    #     kwargs = dict(extra_body={"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": 1024*8}},
    #                   min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0)
    if "gemma-4-26b-a4b" in args.model.lower():
        kwargs = dict(extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    elif "gemma-4" in args.model.lower() and not is_llm_judge_call:
        kwargs = dict(extra_body={"chat_template_kwargs": {"enable_thinking": True}})
    if args.model.startswith("hosted_vllm/"):
        kwargs["base_url"] = args.base_url
        if args.api_key:
            kwargs["api_key"] = args.api_key

    temperature_adder = 0.1 
    temperature = 0
    for _ in range(1, 4):
        content = ""
        try:
            time.sleep(args.llm_call_sleep)
            if args.use_separated_judge and is_llm_judge_call:
                response = litellm.completion(
                    timeout=200,
                    api_key=args.sub_api_key,
                    base_url=args.sub_base_url,
                    model=args.sub_model,
                    messages=messages,
                    temperature=temperature
                )
            elif args.use_separated_summarizer and is_compaction_call:
                response = litellm.completion(
                    timeout=200,
                    api_key=args.sub_api_key,
                    base_url=args.sub_base_url,
                    model=args.sub_model,
                    messages=messages,
                    temperature=temperature
                )
            else:
                response = litellm.completion(
                    timeout=args.llm_call_timeout,
                    model=args.model,
                    messages=messages,
                    # reasoning_effort=args.reasoning_effort,
                    # max_completion_tokens=1024,
                    # reasoning_effort="medium"
                    temperature=temperature,
                    **kwargs
                )

            output_message = response["choices"][0]["message"].to_dict()
            # save LLM calls for further use.
            if args.save_llm_calls and fout and not (is_llm_judge_call or is_compaction_call):
                llm_calls = {"input": messages, "output": output_message, "kwargs": kwargs}
                fout.write(json.dumps(llm_calls) + "\n")
                fout.flush()

            content = output_message["content"]
            if "reasoning" in output_message: # Qwen3 reasoning parser
                reasoning_content = output_message["reasoning"]
            else:   # default reasoning parser (e.g., gpt-oss)
                reasoning_content = output_message.get("reasoning_content", "")
            if "</think>" in content:
                reasoning_in_main_response, content = content.split("</think>", 1)
                reasoning_content += reasoning_in_main_response + "</think>"
            # print_debug(f"LLM response's content: {content}")
            content = parser(dspy_simple_format_correction(content.strip()))

            return (content, reasoning_content) if get_thinking_tokens else content

        except Exception as e:
            print_debug(f"LLM call error: {e}")
            print_debug(f"LLM call re-trying {_}-th time ...")
            time.sleep(2)
            temperature += temperature_adder


# --------------------
# Search Actions
# --------------------

# form a query based on the current search and reasoning state
next_search_query_description = f"""
You are an expert in information retrieval and question answering.
You are doing iterative search and reasoning to find a unique correct answer of a question.

At the moment, you are given 
- a question, 
- some candidate answers that you have produced based on the retrieved evidences so far,
- the history of the search and reasoning process, which consists of status descriptions for each iteration,
- the queries that have been searched, 
- the evidences that have been retrieved based on those queries, and 
- a list of available retrievers (with different search capabilities) to perform the next query.

Your task is to generate the next search query and select which retriever(s) to use for the next search iteration.

**Instructions:**
1. Current status:
    - Focus on the last item in the search and reasoning history, which describes the current status of the search and reasoning process.
    - It expresses what information is still needed and/or why it's important for answering the question and verifying the candidates, or even what is tricky or difficult, etc.
    - Use this information as well as candidate answers to guide the generation of the next specific query and selection of retriever(s).
2. Search strategy:
    - Don't search with a complex query, as it often makes the search less effective.
    - Instead, break down the information need into simple and focused queries, based on the current status and what has been already searched and retrieved.
    - Normally, you should search with short query that targets building-block information to unveil the answer step by step.
        - For example, if the question is "Find X among Y while having Z ...", then search with Y first (broadest scope), then more searches with more specific keywords or answer candidates later.
    - If retrieved evidences from your previous queries don't help much, try a different query that can retrieve different information. Think CREATIVELY.
    - Don't repeat the same query that has been already searched, as it won't retrieve new information.
3. Efficiency:
    - Try to use as few search iterations as possible to find the necessary information to answer the question, by generating effective queries and selecting proper retriever(s) for each query.
4. Retriever selection:
    - Different retrievers have different capabilities and are good at retrieving different types of information.
    - Select the retriever(s) that are most likely to retrieve the information needed for the next query based on their capabilities and the current search and reasoning state.
    - Only use web search when necessary. Try your best to utilize local search indexes.
5. Time-sensitive questions:
    - If the question doesn't specify a specific timestamp (e.g., year 2025) but seems to be time-sensitive, you should infer a specfic time frame (date, month, or even just year) and form the query with that specific time frame based on today's date.
    - FYI, today is in {current_month}.
"""
next_search_query_description_wordcount = len(next_search_query_description.split())

class NextSearchQuery(dspy.Signature):
    # Instruction
    __doc__ = next_search_query_description
    # Input
    question: str = dspy.InputField(desc='The original question to be answered')
    candidate_answers: list[str] = dspy.InputField(desc='The candidate answer(s) to the question that have been produced based on the retrieved evidences so far')
    search_reasoning_history: list[str] = dspy.InputField(desc='The history of the search and reasoning process')
    searched_queries: list[str] = dspy.InputField(desc='The queries which have been searched')
    all_evidences: str = dspy.InputField(desc='Retrieved evidences for the question')
    available_retrievers: list[str] = dspy.InputField(desc='An enumerated list of available retrievers for search, ' \
        'e.g., fuzzy/regex search, dense/sparse search, web search, etc.')
    # Output
    next_query: str = dspy.OutputField(desc='The next search query to be performed')
    retriever_ids: list[int] = dspy.OutputField(desc='The list of ids of retrievers to be used for the next search')


# attempt to answer the question based on the retrieved information
answer_question_description = f"""
You are an expert in question answering and reasoning.
You are doing iterative search and reasoning to find a unique correct answer of a question.

Now, you are given
- a question which has only one correct answer and may be tricky,
- some candidate answers to the question that you suggested in the last iteration, and
- an up-to-date list of retrieved evidences for the question.

Your task is to
- reason about the retrieved evidences
- produce new list of candidate answer(s) (in order of decreasing confidence) to the question with clear reasoning based on the up-to-date list of retrieved evidences, and
- determine the status of the search and reasoning process after this step, e.g.
    - "FINISH" if only *ONE* candidate is produced and you're certain about it, or
    - A concise description of what information is covered by the current evidences,
      what is still needed to answer the question or verify candidates as more search is needed,
      and/or what is tricky or difficult, etc.
- identify which retrieved evidences are certainly not useful or already covered by another more relevant evidence for answering the question.

**Instruction**
1. Evidence reliability:
    - The retrieved evidences can be from various sources with different reliability.
    - You need to consider the attributes of the sources, if provided, along with the content when producing candidates and determining the status.
    - If the source is more reliable, you can be more confident about the information from it.
    - On the other hand, if the source is less reliable, you should be more cautious about the information from it, and try to verify it with more search if it's important for answering the question.
2. Evidence's timeliness for time-sensitive questions:
    - If the question is time-sensitive, the timeliness (e.g., last modification date) of the retrieved evidences is important for producing answer candidates and determining the status. FYI, today is in {current_month}.
    - In this case, you should **always** consider the timestamps of the evidences. If the timestamps are not provided, you can try to infer the timeliness of the evidences based on their content.
    - If the evidences are not timely enough for a time-sensitive question, you should indicate that more search is needed to find more timely evidences.
3. Certainty about the answer:
    - If the evidences are not reliable and/or timely enough though they can be used to answer the question, you should **always** indicate that more search is needed to verify the information and find more reliable and/or timely evidences.
    - Remember that the question only has ONE correct answer, so if you still have multiple candidates, you should indicate that more search is needed to verify which candidate is correct.
    - If you are certain about the answer based on the retrieved evidences, you can produce the final answer and determine the status as "FINISH".
4. Iterative search practice:
    - If you need more search, try to specify what information is still needed and/or why it's important for answering the question and verifying the candidates.
5. **Context efficiency**:
    - For the evidence that you deem not useful or don't use for answering the question, you should identify them to avoid redundant consideration in the future iterations.

**Caution**
1. Be relaxing on the meaning/taxonomies of words/phrases in the question.
    - The question may be tricky and the words/phrases in it may not be meant in their most common way, or it meant to cover a boarder category than their literal meaning.
    - Don't be constrained by the literal meaning of the words/phrases in the question. Instead, try to understand the underlying information need behind the question and answer based on that.
    - Examples,
        - "sound engineer/producer" vs "artist": Nowadays, people often recognize sound engineers as artists.
2. DON'T be affected by reporting biases.
3. Think **CAREFULLY**.
"""
answer_question_description_wordcount = len(answer_question_description.split())

class AnswerQuestion(dspy.Signature):
    # Instruction
    __doc__ = answer_question_description
    # Input
    question: str = dspy.InputField(desc='The question to be answered')
    previous_candidate_answers: list[str] = dspy.InputField(desc='The candidate answer(s) to the question that have been produced in the previous iteration')
    all_evidences: str = dspy.InputField(desc='An enumerated up-to-date list of retrieved evidences for the question')
    # Output
    reasoning: str = dspy.OutputField(desc='The reasoning process and rationale for producing the candidates and determining the status, based on retrieved evidences')
    candidates: list[str] = dspy.OutputField(desc='The new list of candidate answer(s) to the question based on the up-to-date list of retrieved evidences')
    status: str = dspy.OutputField(desc='The status of the search and reasoning process after this step, ' \
        'e.g. "FINISH" if a final answer is produced, a concise description if more search is needed, etc.')
    confidence: float = dspy.OutputField(desc='The confidence score (0-100) if the ground-truth answer is among the candidates, where 100 means very certain and 0 means very uncertain')
    not_useful_evidence_ids: list[int] = dspy.OutputField(desc='The list of ids of evidences that are deemed not useful for answering the question')


final_attempt_additional_instruction = """
**Additional Instruction for Final Attempt**:
Note that this is the FINAL iteration. Please attempt to produce the final answer if you haven't yet, even if you're not very certain about it.
If you still can't produce any candidate, please produce your best guess as the candidate answer. Don't leave it empty.
"""

class AnswerQuestionFinalAttempt(AnswerQuestion):
    __doc__ = answer_question_description + "\n\n" + final_attempt_additional_instruction


# compact and rewrite evidences to keep only relevant parts, while preserving tables and key information
compact_evidences_description = f"""
You are an expert in information extraction and summarization.

You are given
- a question which has only one correct answer, and
- a list of retrieved evidences for the question.

Your task is to summarize relevant and useful information from the list of evidences into a single piece for answering the question.

**Instructions:**
1. For each evidence, determine if it contains information relevant to answering the question.
2. If the evidence is not relevant at all, then ignore that evidence.
3. If the evidence contains relevant information, add the information to the summary:
   - Try to extractive summarization to keep only the most relevant parts
   - Preserves tables and structured data intact (they often contain key facts)
   - Removes redundant, verbose, or irrelevant surrounding content
   - Keeps the core factual information needed to answer the question
4. Reorganize the information in the summary to make it concise and coherent, while keeping the key information intact. There is no hard-limit on the length of the summary, but it should be as concise with no information duplication as possible.
"""

class CompactEvidences(dspy.Signature):
    __doc__ = compact_evidences_description
    question: str = dspy.InputField(desc='The question to be answered')
    all_evidences: list[str] = dspy.InputField(desc='An enumerated list of retrieved evidences for the question')
    compacted_evidence: str = dspy.OutputField(desc='A single, concise summary of relevant and useful information from the list of evidences for answering the question')
    # Compacted versions of the evidences, same length as input — each element is the compacted string, or empty string if the evidence should be skipped


compact_evidence_description = f"""
You are an expert in information extraction and summarization.

You are given
- a question which has only one correct answer, and
- an evidence for the question.

Your task is to compact the evidence to keep only the most relevant parts for answering the question.

**Instructions:**
1. Determine if it contains information relevant to answering the question.
2. If the evidence is not relevant at all, output an empty string for that evidence.
3. If the evidence contains relevant information, produce a compacted version that:
   - Preserves tables and structured data intact (they often contain key facts)
   - Abstractively summarizes other free-form content to keep only relevant parts, ideally within **100 words**
   - Removes redundant, verbose, or irrelevant surrounding content
   - Keeps the core factual information needed to answer the question
4. If the evidence is fully relevant that removing any sentence or part would cause the loss of key information for answering the question:
    - Return "INTACT" to indicate that the evidence should be kept intact without any removal
    - DON'T return the original evidence in this case.
"""
compact_evidence_description_wordcount = len(compact_evidence_description.split())

class CompactEvidence(dspy.Signature):
    __doc__ = compact_evidence_description
    question: str = dspy.InputField(desc='The question to be answered')
    evidence: str = dspy.InputField(desc='An evidence for the question, which can contain long paragraphs or tables or a combination of them')
    compacted_evidence: str = dspy.OutputField(desc='Compacted version of the evidence, which is either "INTACT" (if no change), empty string (if not relevant), or a compacted string with only the relevant parts')
