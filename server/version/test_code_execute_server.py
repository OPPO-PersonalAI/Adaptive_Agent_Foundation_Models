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
Quick test for CodeExec server

Usage:
python test_code_execute_server.py <endpoint_url>
Example: python test_code_execute_server.py http://127.0.0.1:9006/code_exec
"""

import argparse
import json
import requests

def test_code_exec_service():
    parser = argparse.ArgumentParser(description='CodeExec server test script.')
    parser.add_argument('endpoint_url', type=str, help='The endpoint URL to test against (e.g., http://127.0.0.1:9006/code_exec)')
    args = parser.parse_args()

    # Test data
    test_cases = [
        {
            "code_str_list": [
'''
```python
def add(a, b):
    return a + b

result = add(2, 3)
print("[OUTPUT]:", result)
```
''',
'''
```python
def add(a, b):
    return a + b

result = add(-5, -6)
print("[OUTPUT]:", result)
```
'''
            ],
            "parameter_list": [],
        },
        {
            "code_str_list": [
'''
```python
def div(a, b):
    return a / b

result = div(1, 0)
print("[OUTPUT]:", result)
```
'''
            ],
            "parameter_list": [],
        }
    ]

    for data in test_cases:
        print("\n" + "="*20)
        print("="*20)
        try:
            url = args.endpoint_url
            response = requests.post(
                url,
                json=data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            try:
                result = response.json()
                print("\nServer response:")
                print("-" * 50)
                print(result)
                print("-" * 50)
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing error: {e}")
                print(f"Response content: {response.text}")
                continue

            if result.get("success"):
                print("✅ Success!")
                print(f"Processing time: {result.get('processing_time'):.1f} seconds")
                print("\nCode execution result:")
                print("-" * 50)
                print(result.get('obs'))
                print("-" * 50)
            else:
                print(f"❌ Failed: {result.get('error_message', 'Unknown error')}")
        except requests.exceptions.ConnectionError:
            print("❌ Connection error: Cannot connect to server, please make sure the server is running")
            break
        except requests.exceptions.Timeout:
            print("❌ Timeout error: Request timed out")
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {str(e)}")
        except Exception as e:
            print(f"❌ Unknown error: {str(e)}")

if __name__ == '__main__':
    test_code_exec_service()
