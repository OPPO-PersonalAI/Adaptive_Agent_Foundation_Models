import random
import json
import os
import requests


#######################################################################
def CodeExecutorTool(code_exec_url, code, parameter_list=None):
    # Prepare request
    code_str_list = [str(code).strip()]
    if not parameter_list:
        parameter_list = [[]]

    data = {
        "code_str_list": code_str_list,
        "parameter_list": parameter_list,
    }
    # POST to server
    try:
        response = requests.post(
            code_exec_url,
            json=data,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        try:
            result = response.json()
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Response content: {response.text}")

        if result.get("success"):
            print("✅ Success!")
            print(f"Processing time: {result.get('processing_time'):.1f} seconds")
            print("\nResults:")
            print("-" * 50)
            print(result.get('obs'))
            print("-" * 50)
        else:
            print(f"❌ Failed: {result.get('error_message', 'Unknown error')}")
    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Unable to connect to server, please ensure the server is running")

    except requests.exceptions.Timeout:
        print("❌ Timeout error: Request timed out")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {str(e)}")
    except Exception as e:
        print(f"❌ Unknown error: {str(e)}")

    if result.get("success") and "[FAILED]" not in str(result):
        if isinstance(result, dict):
            obs = result["obs"]
            if isinstance(obs, list):
                obs = str(obs[0][1])  # [0] refers to the first code list, [0][1] refers to the actual code output
        else:
            obs = str(result)
        result_str = f"Code executed successfully.\nExecution output:\n{obs}\n"
    elif result['success'] and "[FAILED]" in str(result):
        if isinstance(result, dict):
            obs = result["obs"]
            if isinstance(obs, list):
                obs = str(obs[0][2])
        else:
            obs = str(result)
        result_str = f"ERROR:Code execution failed.\nError:\n{obs}"
    else:
        result_str = f"ERROR:Code server ERROR."
    return result_str


# ##################################################################################################################
def WebSearchTool(web_search_url,query, topk=5):
    """Send request to Serper proxy and print response."""
    web_search_query = query
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": "",
    }
    if topk > 5:
        topk = 5
    payload = {
        "q": query,
        "num": topk,
    }

    try:
        response = requests.post(web_search_url, headers=headers, json=payload)
        response.raise_for_status()  # Raises an exception if status code is not 2xx
        result = response.json()

    except requests.exceptions.RequestException as e:
        result = f"An error occurred: {e}"
    return result


######################################################################################################
def CrawlPageTool(crawl_page_url, api_key, api_url, model, query,url):
    url=[url]
    think_content = ""
    data = {
        "urls": url,
        # "task": task, 
        "web_search_query": query,
        "think_content": think_content,
        "api_url": api_url,
        "api_key": api_key,
        "model": model,
        "summary_prompt_type": "webthinker_with_goal",
        "summary_type": "page",
        # "chunk_size": 8192,
        # "do_last_summary": False
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(crawl_page_url, json=data, timeout=500, headers=headers)
    result = response.json()
    if result.get("success"):
        crawl_page_result = result["obs"]
    else:
        crawl_page_result = result.get("error_message")
    return crawl_page_result
