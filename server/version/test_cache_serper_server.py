#!/usr/bin/env python3
"""
Usage:
python test_cache_serper_server.py <endpoint_url>
e.g. python test_cache_serper_server.py http://127.0.0.1:9002/search
e.g. python test_cache_serper_server.py http://10.236.17.172:9002/search
"""

import argparse
import json
import os
import time

import requests

def test_serper_proxy(endpoint_url: str, query: str, num: int = 10, use_crawl: bool = False, think_content: str = "", web_search_query: str = "", summary_type: str = "once", summary_prompt_type = "webthinker_with_goal"):
    """Send request to Serper proxy and print response."""
    api_url = os.environ.get("SUMMARY_OPENAI_API_BASE_URL")
    api_key = os.environ.get("SUMMARY_OPENAI_API_KEY")
    model = os.environ.get("SUMMARY_MODEL")
    serper_key = os.environ.get("WEB_SEARCH_SERPER_API_KEY")
    jina_key = os.environ.get("JINA_API_KEY")

    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "q": query,  
        "num": num,
    }

    print(f"--- Sending request for query: '{query}' ---")
    start_time = time.time()
    url = endpoint_url
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise exception if status code is not 2xx

        elapsed_time = time.time() - start_time
        print(f"Status Code: {response.status_code} (Response time: {elapsed_time:.2f}s)")
        
        # Print result summary
        result = response.json()
        print(result)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Serper Cache Server")
    parser.add_argument(
        "endpoint_url", 
        type=str, 
        help="The Serper proxy endpoint URL (e.g. http://127.0.0.1:9002/search)"
    )
    args = parser.parse_args()
    print("=== TEST: Search ===")
    test_serper_proxy(
        endpoint_url=args.endpoint_url,  # 读取命令行传入的地址
        query="dog|cat",  # 搜索关键词（可自定义，如"AI latest news"）
        num=10  # 返回10条结果
    )

