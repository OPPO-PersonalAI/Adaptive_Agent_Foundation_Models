import json
import re
import time
import random
import os
random.seed(1234)
import argparse
from openai import OpenAI
from queue import Queue
import logging
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer

from tools import (
    WebSearchTool as RemoteWebSearchTool,
    CrawlPageTool as RemoteCrawlPageTool,
    CodeExecutorTool as RemoteCodeExecutorTool
)
from utils import (
    read_jsonl,
    write_jsonl,
    read_json,
    write_json,
    count_tokens,
    extract_specific_tag_parallel_toolcalldict
)
from prompts import (
    afm_sys_prompt,
    llm_judge_prompt
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B", trust_remote_code=True)
except Exception as e:
    tokenizer = AutoTokenizer.from_pretrained(
        "tokenizer_file",
        trust_remote_code=True)

def get_required_env(key):
    """Get required environment variable, raise error if not found."""
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Required environment variable '{key}' is not set. Please set it before running the script.")
    return value


# Load required environment variables
MODEL = get_required_env("MODEL_NAME")
MODEL_URL = get_required_env("MODEL_URL")
OPENAI_API_URL = get_required_env("OPENAI_API_URL")
OPENAI_API_KEY = get_required_env("OPENAI_API_KEY")
WEBSEARCH_URL = get_required_env("WEBSEARCH_URL")
CRAWL_PAGE_URL = get_required_env("CRAWL_PAGE_URL")
CODE_EXEC_URL = get_required_env("CODE_EXEC_URL")

# Create configurations
URL_CONFIG = {
    "config": [MODEL_URL],
    "pointer": 0,
}

KEY = "empty"
SYSTEM_PROMPT = afm_sys_prompt

## External API ##
#### llm_judge ###
judge_model_config = {
    "model_id": "gpt-5-mini",
    "config": [
        [OPENAI_API_URL, OPENAI_API_KEY],
    ],
    "pointer": 0,
}

## Tool Server ##
### web_search ###
web_search_config = {
    "config": [
        [WEBSEARCH_URL],
    ],
    "pointer": 0,
}

### crawl_page ###
crawl_page_config = {
    "config": [
        [CRAWL_PAGE_URL],
    ],
    "pointer": 0,
}

summary_model_config = {
    "model_id": "gpt-5-mini",
    "config": [
        [OPENAI_API_URL, OPENAI_API_KEY],
    ],
    "pointer": 0
}

### code_execute ###
code_execute_config = {
    "config": [
        [CODE_EXEC_URL]
    ],
    "pointer": 0,
}


def parse_args_and_create_config():
    """Parse command line arguments and create inference configuration for A²FM."""
    parser = argparse.ArgumentParser(description="A²FM: Adaptive Agent Foundation Model Inference")

    # Generation Parameters
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (0.0 to 2.0)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling parameter (0.0 to 1.0)")
    parser.add_argument("--presence_penalty", type=float, default=0,
                        help="Presence penalty (-2.0 to 2.0)")
    parser.add_argument("--frequency_penalty", type=float, default=0,
                        help="Frequency penalty (-2.0 to 2.0)")
    parser.add_argument("--total_tokens", type=int, default=131072,
                        help="Maximum total tokens for generation")

    # Tool Configuration
    parser.add_argument("--web_topk", type=int, default=10,
                        help="Number of top web search results to retrieve")

    # Adaptive Mode Configuration
    parser.add_argument("--adaptive", type=str, default="auto",
                        choices=["auto", "toolcalling_agent", "reasoning_agent", "instant"],
                        help="Adaptive mode selection: 'auto' (automatic mode selection), "
                             "'toolcalling_agent' (force agentic mode with tools), "
                             "'reasoning_agent' (force reasoning mode), "
                             "'instant' (force instant mode for simple tasks)")

    # Max Steps Configuration (renamed from retry_attempts)
    parser.add_argument("--max_steps_agent", type=int, default=60,
                        help="Maximum steps for agentic mode execution")
    parser.add_argument("--max_steps_reasoning", type=int, default=6,
                        help="Maximum steps for reasoning mode execution")
    parser.add_argument("--max_steps_instant", type=int, default=6,
                        help="Maximum steps for instant mode execution")

    # Parallel Processing
    parser.add_argument("--parallel_per_dataset", type=int, default=5,
                        help="Number of parallel processes per dataset")

    # Round Configuration
    parser.add_argument("--round", type=int, default=1,
                        help="Total number of inference rounds")
    parser.add_argument("--start_round", type=int, default=0,
                        help="Starting round number")

    # Data Configuration
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input JSON/JSONL file path")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSONL file path")

    # Parse arguments
    args = parser.parse_args()

    # Create inference kwargs from arguments
    infer_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "presence_penalty": args.presence_penalty,
        "frequency_penalty": args.frequency_penalty,
        "total_tokens": args.total_tokens,
        "web_topk": args.web_topk,
        "max_steps_agent": args.max_steps_agent,
        "max_steps_reasoning": args.max_steps_reasoning,
        "max_steps_instant": args.max_steps_instant,
        "parallel_per_dataset": args.parallel_per_dataset,
        "adaptive": args.adaptive,
        "judge_model_config": judge_model_config,
        "web_search_config": web_search_config,
        "crawl_page_config": crawl_page_config,
        "round": args.round,
        "start_round": args.start_round,
    }

    return args, infer_kwargs


def strong_api_client(system, prompt, url, key, model):
    """
    Client for calling LLM API in "LLM as Judge" scenarios.

    Sends a streaming chat completion request to an OpenAI-compatible API,
    aggregates response chunks, and returns the full judgment result.

    Parameters:
        system (str): LLM judge's role/instructions
        prompt (str): Content to be evaluated/judged
        url (str): API base URL
        key (str): API authentication key
        model (str): Name of the LLM model to use

    Returns:
        str: Complete LLM judgment response

    Exceptions:
        ValueError: Raised on API request failure with error logged
    """
    client = OpenAI(base_url=url, api_key=key)
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        collected_content = []
        for chunk in stream:
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content is not None:
                    content = delta.content
                    collected_content.append(content)
        return "".join(collected_content)
    except Exception as e:
        logging.error(f"[strong_api_client]: {e}")
        time.sleep(1)
        raise ValueError(e)


def test_api_client(system, prompt, current_cls, current_answer, url, key, model, stop_words=None, **kwargs):
    """
    Calls a deployed LLM model to generate output with token budget management and streaming support.

    Sends a chat completion request to an OpenAI-compatible API, handles conversation history,
    enforces token limits, and stops generation at specified stop words. Returns generated content
    along with the stop tag that terminated the generation.

    Parameters:
        system (str): System instructions for the model
        prompt (str): User prompt to guide generation
        current_cls (str): Existing classification/content from previous interaction
        current_answer (str): Existing answer from previous inference
        url (str): API base URL of the deployed model
        key (str): API authentication key
        model (str): Name of the deployed model to use
        stop_words (list, optional): Words/phrases that trigger generation stop.
            Defaults to ["</tool_call>", "</answer>"]
        **kwargs: Additional parameters (timeout, total_tokens, temperature, etc.)

    Returns:
        tuple: (stop_tag, content) where:
            - stop_tag (str): Tag that stopped generation ("error" if failed)
            - content (str): Generated text (with stop tag appended if applicable)
    """
    if stop_words is None:
        stop_words = [
            "</tool_call>",
            "</answer>"
        ]

    client = OpenAI(base_url=url, api_key=key, timeout=kwargs.get("timeout", 1200))

    try:
        # token budget
        system_token_count = count_tokens(system, tokenizer)
        prompt_token_count = count_tokens(prompt, tokenizer)
        current_cls_token_count = count_tokens(current_cls, tokenizer)
        current_answer_token_count = count_tokens(current_answer, tokenizer)
        max_completion_tokens_for_answer = (
                kwargs.get("total_tokens", 32768)
                - system_token_count
                - prompt_token_count
                - current_cls_token_count
                - current_answer_token_count
                - 128
        )

        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": f"user\n\n{system}\n\n{prompt}"},
        ]
        if current_cls and current_answer:
            messages.append({"role": "assistant", "content": f"{current_cls}\n{current_answer}"})
        elif current_cls:
            messages.append({"role": "assistant", "content": current_cls})

        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            stop=stop_words,
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            presence_penalty=kwargs.get("presence_penalty", 0),
            frequency_penalty=kwargs.get("frequency_penalty", 0),
            max_tokens=max_completion_tokens_for_answer,
        )

        collected = []
        stop_tag = None

        for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            if delta is not None:
                piece = getattr(delta, "content", None)
                if piece:
                    collected.append(piece)
            fr = getattr(choice, "finish_reason", None)
            if fr is not None:
                if hasattr(choice, "stop_reason"):
                    stop_tag = getattr(choice, "stop_reason")
                if stop_tag is None and hasattr(choice, "model_extra"):
                    extra = getattr(choice, "model_extra")
                    if isinstance(extra, dict):
                        stop_tag = extra.get("stop_reason")
        text = "".join(collected)
        if stop_tag:
            tag_name = stop_tag.lstrip("</").rstrip(">")
            return tag_name, text + stop_tag
        else:
            return "error", "Cannot find stop_tag"
    except Exception as e:
        return "error", str(e)


def get_toolcall_results_with_format(response,**kwargs):
    """
    Extracts tool call from response, executes corresponding tools,
    and returns aggregated tool execution results with retry logic for failures.

    Parameters:
        response (str): Input text containing tool call tags (parsed by extract_specific_tag_parallel_toolcalldict)
        **kwargs: Optional parameters, including "web_topk" (max results for web search, default 5)

    Returns:
        tuple: (success_status, result_content) where:
            - success_status (bool): True if at least one tool call succeeds; False if all fail or exceptions occur
            - result_content (str): Aggregated tool execution results (if success) or error message (if failure)
    """
    try:
        _, _, tool_calls, parsed_content = extract_specific_tag_parallel_toolcalldict(response)
        web_topk = kwargs.get("web_topk", 5)

        # Get config info
        cur_web_search_config = web_search_config["config"][
            web_search_config["pointer"] % len(web_search_config["config"])]
        cur_crawl_page_config = crawl_page_config["config"][
            crawl_page_config["pointer"] % len(crawl_page_config["config"])]
        cur_summary_model_config = summary_model_config["config"][
            summary_model_config["pointer"] % len(summary_model_config["config"])]
        cur_code_execute_config = code_execute_config["config"][
            code_execute_config["pointer"] % len(code_execute_config["config"])]
        summary_model_id = summary_model_config['model_id']

        # update config pointer
        web_search_config["pointer"] += 1
        crawl_page_config["pointer"] += 1
        summary_model_config["pointer"] += 1
        code_execute_config["pointer"] += 1

        tool_calls_obs = []

        if tool_calls == "tool_call":
            for tool in parsed_content:
                if tool['type'] == 'web_search':
                    logging.info(f"Calling tools: {tool['type']}")
                    query = tool['query']
                    num = web_topk
                    web_results = ""

                    for _ in range(3):
                        web_results = RemoteWebSearchTool(
                            cur_web_search_config[0],
                            query=query,
                            topk=num
                        )
                        if "BudgetExceededError" in str(web_results) or "An error occurred:" in str(
                                web_results) or "400 Bad Request" in str(web_results):
                            logging.info("Error in web search's observation, retrying...")
                            web_results = "web search server error"
                            time.sleep(0.1)
                        else:
                            break

                    if web_results:
                        tool["obs"] = web_results
                        tool_calls_obs.append(tool)
                        logging.info(f"Tool calls done: {tool['type']}")
                    else:
                        logging.info("Failed after retrying web search for 3 times!")

                elif tool['type'] == "crawl_page":
                    logging.info(f"Calling tools: {tool['type']}")
                    url = tool['url']
                    query = tool['query']

                    for _ in range(3):
                        crawl_results = RemoteCrawlPageTool(
                            crawl_page_url=cur_crawl_page_config[0],
                            api_key=cur_summary_model_config[1],
                            api_url=cur_summary_model_config[0],
                            model=summary_model_id,
                            query=query,
                            url=url,
                        )
                        if "BudgetExceededError" in str(crawl_results) or "An error occurred:" in str(
                                crawl_results) or "400 Bad Request" in str(crawl_results):
                            logging.info("Error in crawl page's observation, retrying...")
                            crawl_results = "crawl_page error"
                            time.sleep(0.1)
                        else:
                            break

                    if crawl_results:
                        tool["obs"] = crawl_results
                        tool_calls_obs.append(tool)
                        logging.info(f"Tool calls done: {tool['type']}")
                    else:
                        logging.info("Failed after retrying crawl page for 3 times!")

                elif tool['type'] == "code_execute":
                    logging.info(f"Calling tools: {tool['type']}")
                    code = tool['code']

                    for _ in range(3):
                        code_results = RemoteCodeExecutorTool(
                            code_exec_url=cur_code_execute_config[0],
                            code=code,
                            parameter_list=[]
                        )
                        if "UnboundLocalError" in str(code_results):
                            logging.info("Error in code execute's observation, retrying...")
                            code_results = "code_execute error"
                            time.sleep(0.1)
                        else:
                            break

                    if code_results:
                        tool["obs"] = code_results
                        tool_calls_obs.append(tool)
                        logging.info(f"Tool calls done: {tool['type']}")
                    else:
                        logging.info("Failed after retrying code execute for 3 times!")

                elif tool['type'] == "dummy_tool":
                    logging.info(f"dummy_tool call")
                    tool["obs"] = tool.get("error_msg", "")
                    if not tool["obs"]:
                        continue
                    tool_calls_obs.append(tool)
                    logging.info(f"dummy_tool: {tool['obs']}")

            logging.info(f"Total calls: {len(parsed_content)}, Actual tool responses: {len(tool_calls_obs)}!")
            if not tool_calls_obs:
                return False, "All tool calls failed!"

            tool_obs = ""
            for i, tool in enumerate(tool_calls_obs, start=1):
                tool_obs += f"\nResults for tool call {i}:\n"
                tool_obs += str(tool['obs'])
            return True, tool_obs
    except Exception as err:
        return False, str(err)


def process_single_data(query, **kwargs):
    """
    Core function to process a single task query: classifies the task type, selects the appropriate mode,
    and generates responses/trajectories.

    Parameters:
        query (str): The user's task query to be processed
        **kwargs: Optional configuration parameters including:
            - adaptive (str): Agent selection mode ("auto" for model-classified, "reasoning_agent"/"toolcalling_agent"/"instant" for manual)
            - max_steps_agent (int): Max steps for toolcalling_agent

    Returns:
        tuple: (classification_result, trajectory_list, error_msg, full_content) where:
            - classification_result (str): Selected agent type ("reasoning_agent"/"toolcalling_agent"/"default_agent" or None if failed)
            - trajectory_list (list[dict]): List of generated steps 
            - error_msg (str/None): Error message if max attempts exceeded (None if answer found)
            - full_content (str): Concatenated full content (classification + all generated responses)
    """
    system_prompt = SYSTEM_PROMPT.strip()
    current_answer = ""
    current_cls = ""

    result_list = []
    attempt = 0
    error_count = 0
    if kwargs["adaptive"] == "auto":
        classification_content = None
        while attempt < 10:
            logging.info(f"Starting model classification task: {MODEL}")
            time.sleep(random.random() * 0.1)
            URL = URL_CONFIG["config"][URL_CONFIG["pointer"] % len(URL_CONFIG["config"])]
            URL_CONFIG["pointer"] += 1
            stop_words = "</classification>"
            logging.info(f"Starting classification: {MODEL}")
            tag, content = test_api_client(system_prompt, query, "", "", URL, KEY, MODEL, stop_words=stop_words,
                                           **kwargs)
            current_cls = content
            _, _, _, classification_content = extract_specific_tag_parallel_toolcalldict(content, allowed_tags=[
                "classification"])
            logging.info(f"Classification completed: {MODEL}, classification result: {classification_content}")
            if tag == "error" or content is None or classification_content not in ["toolcalling_agent",
                                                                                   "reasoning_agent", "default_agent"]:
                logging.info("Model classification error!!")
                error_count += 1
                continue
            break
    elif kwargs["adaptive"] == "reasoning_agent":
        classification_content = "reasoning_agent"
        current_cls = "This task requires complex logical reasoning (such as mathematical proofs, multi-step problem solving) and causal analysis, so I will select reasoning_agent.\n<classification>\nreasoning_agent\n</classification>\n"
    elif kwargs["adaptive"] == "toolcalling_agent":
        classification_content = "toolcalling_agent"
        current_cls = "This task requires acquiring real-world information (such as news and data) or executing code (such as programming problems, data processing, or statistics), so I will select toolcalling_agent.\n<classification>\ntoolcalling_agent\n</classification>\n"
    elif kwargs["adaptive"] == "instant":
        classification_content = "default_agent"
        current_cls = "This task needs no real-world info, code, or complex reasoning—just basic knowledge or brief responses, so I will select default_agent.\n<classification>\ndefault_agent\n</classification>\n"

    if classification_content == "reasoning_agent":
        retry_attempts = kwargs["max_steps_reasoning"]
        current_answer = f""
    elif classification_content == "toolcalling_agent":
        retry_attempts = kwargs["max_steps_agent"]
        current_answer = f""
    elif classification_content == "default_agent":
        retry_attempts = kwargs["max_steps_instant"]
        current_answer = f""

    while attempt < retry_attempts and error_count < 10:
        time.sleep(0.1)
        URL = URL_CONFIG["config"][URL_CONFIG["pointer"] % len(URL_CONFIG["config"])]
        URL_CONFIG["pointer"] += 1
        logging.info(f"Starting to generate current trajectory: {MODEL}")

        item_type, content = test_api_client(system_prompt, query, current_cls, current_answer, URL, KEY, MODEL,
                                             **kwargs)
        logging.info(
            f"Current trajectory generation completed: {MODEL}, current round response length: {count_tokens(content, tokenizer)}")

        logging.info(f"step {attempt + 1}: {item_type} | {content[:100]}...")
        if item_type == "error" or content is None:
            error_count += 1
            continue
        elif item_type == "answer":
            result_list.append({
                "type": item_type,
                "content": content
            })
            current_answer += content
            attempt += 1
            return classification_content, result_list, None, current_cls + current_answer
        elif item_type in ["tool_call"]:
            logging.info(f"Starting {item_type}")
            current_answer_with_current_content = current_answer + content
            is_success, tool_calls_obs = get_toolcall_results_with_format(content, **kwargs)

            if not is_success:
                logging.warning(f"Error occurred while using tool: {tool_calls_obs}")
                error_count += 1
                continue
            result_list.append({
                "type": item_type,
                "content": f"\n{content}\n\n<tool_response>\n{tool_calls_obs}\n</tool_response>\n\n"
            })
            current_answer += f"\n{content}\n\n<tool_response>\n{tool_calls_obs}\n</tool_response>\n\n"
            attempt += 1

    return classification_content, result_list, "Exceeded max_attempts and no answer found", current_cls + current_answer


def decode_response(response):
    try:
        if isinstance(response, str):
            return json.loads(response)
        return response
    except:
        return {"judgement": "incorrect"}


import random
import os
import logging


def process_queries(infile, outfile, q_key, a_key, **kwargs):
    # Read input data
    if infile.endswith(".json"):
        questions_data = read_json(infile)
    elif infile.endswith(".jsonl"):
        questions_data = read_jsonl(infile)
    else:
        raise ValueError(f"Unsupported file format: {infile}")

    # Check if output file exists and deduplicate
    out_set = set()
    if os.path.exists(outfile):
        out_data = read_jsonl(outfile)
        out_data = [item for item in out_data if item["prediction"] is not None]
        write_jsonl(out_data, outfile)
        out_set = set([item["question"] for item in out_data])

    # Count of processed data
    processed_count = len(out_set)
    logging.info(f"Processed data count: {processed_count}")

    logging.info(outfile)
    new_questions_data = [item for item in questions_data if item[q_key] not in out_set]
    logging.info(f"Initial data: {len(questions_data)}, Filtered new data: {len(new_questions_data)}")
    questions_data = new_questions_data

    # Check if random sampling is needed (ensure total sample size = processed + new samples)
    sample_total = kwargs.get("sample")
    if sample_total is not None and isinstance(sample_total, int) and sample_total > 0:
        # Number of new samples needed = total sample size - already processed count
        need_sample = max(0, sample_total - processed_count)

        if need_sample > 0:
            # Ensure sample count doesn't exceed available new data
            actual_sample = min(need_sample, len(questions_data))
            questions_data = random.sample(questions_data, actual_sample)
            logging.info(
                f"Need to sample {need_sample} new data, actually sampled {actual_sample}, total sample size will reach {processed_count + actual_sample}")
        else:
            # Processed count already meets or exceeds total sample requirement
            questions_data = []
            logging.info(
                f"Processed data count {processed_count} already meets or exceeds total sample requirement {sample_total}, no new data sampling needed")
    else:
        # If no sampling needed, keep the original shuffle operation
        random.shuffle(questions_data)
        logging.info(
            f"No valid sampling parameter specified, using all filtered data, total {len(questions_data)} items")

    # Initialize statistics and shared queues
    stats = {"total": len(questions_data), "success": 0, "failed": 0}
    task_queue = Queue()
    result_queue = Queue()
    write_lock = Lock()  # Lock for file writing

    # Producer function - put tasks into queue
    def producer():
        for idx, question_data in enumerate(questions_data):
            task_queue.put((idx, question_data))
        # Put end markers
        for _ in range(kwargs.get("parallel_per_dataset", 4)):
            task_queue.put(None)

    # Consumer function - get tasks from queue and process
    def consumer():
        cur_judge_model_config_list = random.choice(judge_model_config["config"])
        judge_model_id = judge_model_config["model_id"]

        nonlocal stats
        while True:
            task = task_queue.get()
            if task is None:  # End marker
                break

            idx, question_data = task
            question = question_data[q_key]
            golden_answer = question_data[a_key]
            level = question_data.get('Level', '-1')

            max_retry = 3
            result = 0  # Default to failure
            trace = None

            for retry in range(max_retry):
                trace = {
                    "question_id": str(idx),
                    "question": question,
                    "Level": level,
                    "golden_answer": golden_answer,
                    "prediction": None,
                    "llm_judge": 0,
                    "tag": None,
                    "steps": [],
                    "status": None,
                    "error": None,
                }

                classification_tag, result_list, failed_reason, content = process_single_data(question, **kwargs)

                if failed_reason:
                    trace["status"] = "error"
                    trace["error"] = failed_reason
                elif "BudgetExceededError" in str(result_list):
                    trace["status"] = "error"
                    trace["error"] = "Detected special vocabulary, belongs to tool calling problem"
                else:
                    trace["steps"] = result_list
                    if any([result_dict["type"] == "error" for result_dict in result_list]):
                        trace["status"] = "error"
                        trace["error"] = "Error occurred during processing"
                    else:
                        if result_list[-1]["type"] == "answer":
                            try:
                                prediction = re.findall(r'<answer>(.*?)</answer>',
                                                        result_list[-1]["content"],
                                                        re.DOTALL)[0].strip()
                                trace["prediction"] = prediction
                                trace["status"] = "completed"
                            except (IndexError, AttributeError) as e:
                                trace["status"] = "parse_error"
                                trace["error"] = f"Failed to parse answer: {str(e)}, {result_list[-1]['content']}"
                        else:
                            trace["status"] = "invalid_format"
                            trace["error"] = f"Last step is not final_tag type"

                # If no error, perform LLM evaluation
                if not trace["error"]:
                    llm_evaluation_prompt = llm_judge_prompt.format(
                        question=question,
                        gt_answer=golden_answer,
                        pred_answer=trace["prediction"]
                    )
                    output = strong_api_client(
                        "You are an evaluation assistant.",
                        llm_evaluation_prompt,
                        cur_judge_model_config_list[0],
                        cur_judge_model_config_list[1],
                        judge_model_id
                    )
                    json_output = decode_response(output)
                    if (json_output and isinstance(json_output, dict) and
                            "judgement" in json_output and
                            json_output['judgement'].lower() == "correct"):
                        trace['llm_judge'] = 1
                    else:
                        trace['llm_judge'] = 0
                    logging.info("#" * 50)
                    logging.info(f"-- Level: {trace['Level']}")
                    logging.info(f"-- question: {question}")
                    logging.info(f"-- predicted_answer: {trace['prediction']}")
                    logging.info(f"-- golden_answer: {golden_answer}")
                    logging.info(f"-- llm_judge: {trace['llm_judge']}")
                    logging.info(f"-- classification_tag: {classification_tag}")
                    logging.info(f"Successfully processed query {idx}, status: {trace['status']}")
                    logging.info("#" * 50)

                    trace["raw"] = question_data
                    trace["all_content"] = content
                    trace["tag"] = classification_tag
                    result = 1  # Success count
                    break  # Exit retry loop if successful
                logging.info(f"ERROR: Query {idx} processing failed, reason: {trace['error']}")

            # Put result in result queue
            result_queue.put((result, trace))
            task_queue.task_done()
        return

    def result_writer():
        nonlocal stats
        # First read existing data
        existing_data = []
        if os.path.exists(outfile):
            existing_data = read_jsonl(outfile)

        while True:
            result_item = result_queue.get()
            if result_item is None:  # End marker
                break
            result, trace = result_item
            # Only process when prediction is not None
            if trace.get("prediction") is not None:
                if result == 1:
                    stats["success"] += 1
                else:
                    stats["failed"] += 1
                # Append to existing data
                existing_data.append(trace)
                # Only lock when writing, ensure atomicity of each write
                with write_lock:
                    # Write to file (append mode)
                    write_jsonl([trace], outfile, "a")
            else:
                # Cases where prediction is None are not counted in statistics
                logging.info(f"Skip writing: prediction is None (question: {trace.get('question')})")
            result_queue.task_done()
        return

        # Create thread pool

    num_workers = kwargs.get("parallel_per_dataset", 4)
    with ThreadPoolExecutor(max_workers=num_workers + 1) as executor:
        # Start producer thread
        executor.submit(producer)
        # Start consumer threads
        consumer_futures = [executor.submit(consumer) for _ in range(num_workers)]
        # Start result writer thread
        writer_future = executor.submit(result_writer)
        # Wait for all tasks to complete
        for future in as_completed(consumer_futures):
            future.result()
        # After all consumers are done, send end signal to writer thread
        result_queue.put(None)
        writer_future.result()

    # Save statistics
    stats_file = outfile.replace(".jsonl", ".param_stats.json")
    write_json({**kwargs, **stats}, stats_file)
    logging.info(
        f"Processing complete! Success: {stats['success']}, Failed: {stats['failed']}, Total: {len(new_questions_data)}")
    return outfile


def ensure_directory_exists(file_path):
    """Ensure the directory for the file exists, create it if not"""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")


def process_single_dataset(infile, outfile_base, q_key, a_key, **infer_kwargs):
    """Function to process a single dataset"""
    start_time = time.time()
    all_outfiles = []

    # Ensure output directory exists
    ensure_directory_exists(outfile_base)

    # Generate all output files for current dataset and process
    for current_round in range(infer_kwargs["start_round"], infer_kwargs["round"]):
        current_outfile = outfile_base.replace(".jsonl", f".round_{current_round}.jsonl")
        all_outfiles.append(current_outfile)
        # Ensure directory exists for each output file
        ensure_directory_exists(current_outfile)
        # Call processing function
        process_queries(infile, current_outfile, q_key, a_key, **infer_kwargs)

    # Analyze results
    stats_file = outfile_base.replace(".jsonl", ".output_stats.txt")
    bad_case_file = outfile_base.replace(".jsonl", ".bad_case.jsonl")

    # Ensure directories exist for stats and bad case files
    ensure_directory_exists(stats_file)
    ensure_directory_exists(bad_case_file)

    cost_time = time.time() - start_time
    logging.info(f"Dataset {infile} processing complete, time taken: {cost_time:.2f} seconds")
    return infile, cost_time


def main():
    # Parse command line arguments and create configuration
    args, INFER_KWARGS = parse_args_and_create_config()

    # Create show kwargs for display
    SHOW_KWARGS = {
        "key": KEY,
        "model": MODEL,
        "url_config": URL_CONFIG,
        "system_prompt": SYSTEM_PROMPT,
        **INFER_KWARGS
    }

    # Display configuration information
    logging.info("=" * 50)
    logging.info("A²FM Inference Configuration")
    logging.info("=" * 50)
    for key, value in SHOW_KWARGS.items():
        if key not in ["judge_model_config", "web_search_config", "crawl_page_config"]:
            logging.info(f">>>> {key}: {value}")

    logging.info("=" * 50)
    logging.info(f"Input file: {args.input_file}")
    logging.info(f"Output file: {args.output_file}")
    logging.info("=" * 50)

    # Define configurations for single dataset
    datasets = [
        {
            "infile": args.input_file,
            "q_key": "question",
            "a_key": "answer",
            "outfile_base": args.output_file
        },
    ]

    # Record total start time
    total_start_time = time.time()
    # Process single dataset
    for dataset in datasets:
        try:
            dataset_path, cost_time = process_single_dataset(
                dataset["infile"],
                dataset["outfile_base"],
                dataset["q_key"],
                dataset["a_key"],
                **INFER_KWARGS,
            )
            logging.info(f"Dataset {dataset_path} processing complete, time taken: {cost_time:.2f} seconds")
        except Exception as e:
            logging.error(f"Error occurred while processing dataset: {str(e)}")

    # Calculate total time
    total_cost_time = time.time() - total_start_time
    logging.info(f"Single dataset processing complete, total time taken: {total_cost_time:.2f} seconds")
    return


if __name__ == "__main__":
    main()
