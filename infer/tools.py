import random
import json
import os
import requests
def CodeExecutorTool(code_exec_url, code, parameter_list=None):
    """
    A tool for executing code by sending it to a remote code execution server via HTTP POST request.
    
    This function packages the provided code and parameters into a JSON payload,
    sends it to the specified code execution endpoint, handles potential errors during
    the request and response processing, and returns a formatted result string.
    
    Parameters:
        code_exec_url (str): URL of the remote code execution server endpoint
        code: The code to be executed (will be converted to string)
        parameter_list (list, optional): List of parameters to accompany the code execution.
            Defaults to an empty list if not provided.
    
    Returns:
        str: Formatted result string indicating success or failure of code execution,
            including relevant output or error messages
    """
    # Prepare request data structure
    code_str_list = [str(code).strip()]
    if not parameter_list:
        parameter_list = []

    data = {
        "code_str_list": code_str_list,
        "parameter_list": parameter_list,
    }
    
    # Send POST request to the code execution server
    try:
        response = requests.post(
            code_exec_url,
            json=data,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse JSON response from server
        try:
            result = response.json()
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Response content: {response.text}")

        # Handle successful execution response
        if result.get("success"):
            print("✅ Success!")
            print(f"Processing time: {result.get('processing_time'):.1f} seconds")
            print("\nResults:")
            print("-" * 50)
            print(result.get('obs'))
            print("-" * 50)
        else:
            print(f"❌ Failed: {result.get('error_message', 'Unknown error')}")
    
    # Handle different types of request exceptions
    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Unable to connect to server, please ensure the server is running")
    except requests.exceptions.Timeout:
        print("❌ Timeout error: Request timed out")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {str(e)}")
    except Exception as e:
        print(f"❌ Unknown error: {str(e)}")

    # Prepare formatted result string based on execution outcome
    if result.get("success") and "[FAILED]" not in str(result):
        if isinstance(result, dict):
            obs = result["obs"]
            if isinstance(obs, list):
                # Extract actual code output from response structure
                obs = str(obs[0][1])  # [0] refers to first code entry, [1] refers to its output
        else:
            obs = str(result)
        result_str = f"Code executed successfully.\nExecution output:\n{obs}\n"
    elif result['success'] and "[FAILED]" in str(result):
        if isinstance(result, dict):
            obs = result["obs"]
            if isinstance(obs, list):
                # Extract error information from response structure
                obs = str(obs[0][2])
        else:
            obs = str(result)
        result_str = f"ERROR:Code execution failed.\nError:\n{obs}"
    else:
        result_str = f"ERROR:Code server ERROR."
    return result_str

def WebSearchTool(web_search_url, query, topk=5):
    """
    A tool to send web search requests to a Serper proxy server and return the search response.
    
    This function constructs a search request with the provided query and result count limit,
    sends it to the specified Serper proxy endpoint, handles potential request errors,
    and returns either the parsed JSON search result or an error message.
    
    Parameters:
        web_search_url (str): URL of the Serper proxy server endpoint to send search requests
        query (str): The search query string (keywords or questions) to retrieve information
        topk (int, optional): Maximum number of top search results to request. 
            Defaults to 5, and will be capped at 5 if a larger value is provided.
    
    Returns:
        Union[dict, str]: Parsed JSON response (as a dictionary) from the Serper proxy if successful;
            otherwise, a string describing the error that occurred.
    """
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

def CrawlPageTool(crawl_page_url, api_key, api_url, model, query, url):
    """
    A tool to crawl a specific web page and process its content using a specified model.
    
    This function sends a request to a web crawling service to retrieve and process
    content from a target URL based on a search query. It integrates with an API service
    using provided credentials and model specifications to generate structured results.
    
    Parameters:
        crawl_page_url (str): URL of the web crawling service endpoint
        api_key (str): Authentication key for accessing the API service
        api_url (str): URL of the API service used for content processing
        model (str): Name/identifier of the model to use for content processing
        query (str): Search query to guide content extraction and processing
        url (str): Target web page URL to be crawled and processed
    
    Returns:
        str: Processed content from the crawled page if successful; 
             otherwise, an error message describing the failure
    """
    url = [url]
    think_content = ""
    data = {
        "urls": url,
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
