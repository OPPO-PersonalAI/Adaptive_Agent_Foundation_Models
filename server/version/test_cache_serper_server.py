#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Inc. Personal AI team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Usage:
python test_cache_serper_server.py <endpoint_url>
e.g. python test_cache_serper_server.py http://127.0.0.1:9002/search
e.g. python test_cache_serper_server.py http://10.236.17.172:9002/search
"""

import argparse
import time
import requests


def test_serper_proxy(endpoint_url: str, query: str, num: int = 10):
    """Send request to Serper proxy and print response."""
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
        endpoint_url=args.endpoint_url,
        query="dog|cat",
        num=10
    )

