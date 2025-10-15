#!/usr/bin/env python3
"""
Quick test for CrawlPage server

Usage:
python test_crawl_page_simple.py <endpoint_url>
e.g. python test_crawl_page_simple.py http://127.0.0.1:9000/crawl_page
"""

import argparse
import json
import os
import time

import requests

# Create argument parser
parser = argparse.ArgumentParser(description='Crawl page test script.')
parser.add_argument('endpoint_url', type=str,
                    help='The endpoint URL to test against (e.g., http://127.0.0.1:9000/crawl_page)')

# Parse command line arguments
args = parser.parse_args()

# Test data
api_url = os.environ.get("SUMMARY_OPENAI_API_BASE_URL")
api_key = os.environ.get("SUMMARY_OPENAI_API_KEY")
model = os.environ.get("SUMMARY_MODEL")

if not all([api_url, api_key, model]):
    print("❌ Error: Environment variables not set correctly")
    print(f"api_url: {api_url}, api_key: {api_key}, model: {model}")
    exit(1)
else:
    print("Using API")
    print(f"api_url: {api_url}")
    print(f"api_key: {api_key}")
    print(f"model: {model}")

# test diff summary type
data_webthinker_with_goal = {
    "urls": ["https://en.wikipedia.org/wiki/Qwen", "https://en.wikipedia.org/wiki/Alibaba_Cloud"],
    "web_search_query": "qwen is developed by?",
    "think_content": "I want to know who delelop qwen?",
    "summary_prompt_type": "webthinker_with_goal",
    "summary_type": "once",
    "api_url": api_url,
    "api_key": api_key,
    "model": model,
    "task": "test",
}

data_webdancer_with_goal = {
    "urls": ["https://en.wikipedia.org/wiki/Qwen", "https://en.wikipedia.org/wiki/Alibaba_Cloud"],
    "task": "qwen is developed by?",
    "web_search_query": "qwen is developed by?",
    "think_content": "to think...",
    "api_url": api_url,
    "api_key": api_key,
    "model": model,
    "summary_prompt_type": "webdancer_with_goal",
    "summary_type": "page",
}

data_webthinker = {
    "urls": ["https://en.wikipedia.org/wiki/Qwen", "https://en.wikipedia.org/wiki/Alibaba_Cloud"],
    "task": "qwen is developed by?",
    "web_search_query": "qwen is developed by?",
    "think_content": "to think...",
    "api_url": api_url,
    "api_key": api_key,
    "model": model,
    "summary_prompt_type": "webthinker",
    "summary_type": "page",
}

data_webdancer = {
    "urls": ["https://en.wikipedia.org/wiki/Qwen", "https://en.wikipedia.org/wiki/Alibaba_Cloud"],
    "task": "qwen is developed by?",
    "web_search_query": "qwen is developed by?",
    "think_content": "to think...",
    "api_url": api_url,
    "api_key": api_key,
    "model": model,
    "summary_type": "once",
    "summary_prompt_type": "webdancer",
}

# all_data = [data_webthinker_with_goal, data_webdancer_with_goal, data_webthinker, data_webdancer]
all_data = [data_webthinker_with_goal]

for data in all_data:
    print("\n" + "=" * 20)
    print(f"Testing Summary Type: {data.get('summary_type')}")
    print(f"Testing Summary Prompt Type: {data.get('summary_prompt_type')}")
    print("=" * 20)
    try:
        # Send request
        url = args.endpoint_url
        response = requests.post(
            url,
            json=data,
            headers={"Content-Type": "application/json"}
        )

        # Check HTTP status code
        response.raise_for_status()

        try:
            result = response.json()
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Response content: {response.text}")  # Print first 500 chars of response
            continue

        # Output results
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
        print("❌ Connection error: Cannot connect to server, please make sure the server is running")
        break  # Stop testing if connection fails
    except requests.exceptions.Timeout:
        print("❌ Timeout error: Request timed out")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {str(e)}")
    except Exception as e:
        print(f"❌ Unknown error: {str(e)}")

print("\nAll tests completed")
