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
import json
import re
from typing import List, Dict, Any, Union
from collections import deque
import re
from collections import deque
import ast


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Read a JSONL (JSON Lines) file and return a list of data
    Each line is parsed into a JSON object, which are combined into a list and returned
    
    Parameters:
        file_path (str): Path to the JSONL file
    
    Returns:
        List[Dict[str, Any]]: A list containing all JSON objects
    
    Exceptions:
        Handle cases where the file does not exist and parsing errors, return an empty list and print a warning
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                try:
                    # Remove leading and trailing whitespace from lines and parse JSON
                    json_obj = json.loads(line.strip())
                    data.append(json_obj)
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse line {line_num}, skipping this entry")
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist")
    except Exception as e:
        print(f"An error occurred while reading the file: {str(e)}")
    return data


def write_jsonl(
    data: List[Dict[str, Any]], 
    file_path: str, 
    append: bool = False, 
    ensure_ascii: bool = False
) -> bool:
    """
    Write the data list to a JSONL file, with one JSON object per line
    Supports append mode and non-ASCII character handling
    
    Parameters:
        data (List[Dict[str, Any]]): The list of JSON objects to be written
        file_path (str): The target file path
        append (bool): Whether to use append mode, default is False (overwrite and write)
        ensure_ascii (bool): Whether to escape non-ASCII characters, default is False (retain characters such as Chinese)
    
    Returns:
        bool: Whether the writing is successful
    
    Example:
        write_jsonl([{"key": "value"}], "data.jsonl")
    """
    try:
        mode = 'a' if append else 'w'
        with open(file_path, mode, encoding='utf-8') as file:
            for item in data:
                # Convert to a JSON string and ensure that non-ASCII characters are correctly encoded
                json_line = json.dumps(item, ensure_ascii=ensure_ascii) + '\n'
                file.write(json_line)
        return True
    except Exception as e:
        print(f"An error occurred while writing to the file：{str(e)}")
        return False


def read_json(file_path: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Read a standard JSON file and return the parsed Python object
    Supports JSON files in object format and array format
    
    Parameters:
        file_path (str): Path to the JSON file
    
    Returns:
        Union[Dict[str, Any], List[Dict[str, Any]]]: The Python object after JSON parsing
    
    Exceptions:
        Handle cases such as non-existent files and parsing errors, return None and print error information
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {str(e)}")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {str(e)}")
        return None


def write_json(
    data: Union[Dict[str, Any], List[Dict[str, Any]]], 
    file_path: str, 
    indent: int = 2, 
    ensure_ascii: bool = False,
    sort_keys: bool = False
) -> bool:
    """
    Write Python objects to a standard JSON file
    Supports data of dictionary and list types
    
    Parameters:
        data (Union[Dict[str, Any], List[Dict[str, Any]]]): The Python object to be written
        file_path (str): Target file path
        indent (int): Number of indentation spaces, default is 2 (for beautiful formatting)
        ensure_ascii (bool): Whether to escape non-ASCII characters, default is False (retains characters such as Chinese)
        sort_keys (bool): Whether to sort by keys, default is False
    
    Returns:
        bool: Whether the writing is successful
    
    Example:
        write_json({"key": "value"}, "data.json")
    """
    try:
        with open(file_path, "w", encoding='utf-8') as file:
            json.dump(
                data, 
                file, 
                indent=indent, 
                ensure_ascii=ensure_ascii,
                sort_keys=sort_keys
            )
        return True
    except Exception as e:
        print(f"An error occurred while writing to the file：{str(e)}")
        return False


def count_tokens(text, tokenizer):
    """Calculate the number of tokens in a text using the transformers tokenizer"""
    if not text:
        return 0
    return len(tokenizer.encode(text))

def extract_specific_tag_parallel_toolcalldict(text, allowed_tags=None):
    """
    Extract specific tags and their content from a text, with special handling for 'tool_call' tags.
    Supports parallel tool calls parsing and error handling for invalid formats.
    
    Parameters:
        text (str): The input text containing tagged segments
        allowed_tags (List[str], optional): List of allowed tag names. 
            Defaults to ['answer', 'plan', 'summary', 'classification', 'reasoning', 'tool_call']
    
    Returns:
        Tuple[Optional[str], Optional[str], str, Union[List[Dict], str]]: A tuple containing:
            - think_content: The content before the last tag (if exists)
            - last_content: Any content after the last closing tag (if exists)
            - tool: The name of the last tag found
            - parsed_content: Parsed content, which is a list of dictionaries for 'tool_call' 
              or a string for other tags
    
    Exceptions:
        Asserts may fail if the text structure doesn't match expected tagged format
    """
    if allowed_tags is None:
        allowed_tags = ['answer', 'plan', 'summary', 'classification', 'reasoning', 'tool_call']

    split_pattern = re.compile(r'(<\/?(?:{})>)'.format('|'.join(allowed_tags)))
    segments = split_pattern.split(text)
    segments = [s for s in segments if s.strip()]

    # Extract information of the last tag and process
    if segments[-1].strip().lstrip("</").rstrip(">").strip() in allowed_tags:
        tool = segments[-1].strip().lstrip("</").rstrip(">").strip()
        content = segments[-2].strip()
        assert tool in segments[-3]
        think_content = segments[-4].strip() if len(segments) > 3 else None
        last_content = None

        # Process 'tool_call' tag, supporting multiple tool calls
        if tool == "tool_call":
            parsed_content = []
            tool_calls = content.split("{'id'")
            new_tool_calls = []
            for tool_call in tool_calls:
                if tool_call:
                    tool_call = "{'id'"+tool_call
                    new_tool_calls.append(tool_call)

            # Clean whitespace from each call and filter empty strings
            tool_calls = [call.strip() for call in new_tool_calls if call.strip()]

            for call_idx, call in enumerate(tool_calls, start=1):
                try:
                    # 1. Attempt to parse the tool call dictionary
                    try:
                        call_dict = ast.literal_eval(call)
                    except (SyntaxError, ValueError) as e:
                        # Parsing syntax error: record specific error information
                        parsed_item = {
                            'type': "dummy_tool",
                            'id': call_idx,  # Generate unique error ID using index
                            'raw': call,
                            'error_msg': f"Parsing arguments failed with error message: {str(e)}, please output arguments in json format string."
                        }
                        parsed_content.append(parsed_item)
                        continue

                    # 2. Check if required keys exist in the dictionary
                    required_keys = ['id', 'name', 'arguments']
                    missing_keys = [k for k in required_keys if k not in call_dict]
                    if missing_keys:
                        parsed_item = {
                            'type': "dummy_tool",
                            'id': call_idx,
                            'raw': call,
                            'error_msg': f"Tool call has no {', '.join(missing_keys)}"
                        }
                        parsed_content.append(parsed_item)
                        continue

                    # 3. Extract basic information
                    tool_type = call_dict['name']
                    call_id = call_dict['id']
                    arguments = call_dict['arguments']

                    # 4. Build basic parsed item (with default error field)
                    parsed_item = {
                        'type': tool_type,
                        'id': call_id,
                        'raw': call,
                        'error_msg': None  # None when no error
                    }

                    # 5. Validate and add parameters based on tool type, record errors for missing parameters
                    if tool_type == 'web_search':
                        query = arguments.get('query', '')
                        parsed_item['query'] = query

                    elif tool_type == 'crawl_page':
                        url = arguments.get('url', '')
                        query = arguments.get('query', '')
                        parsed_item['url'] = url
                        parsed_item['query'] = query


                    elif tool_type == 'code_execute':
                        code = arguments.get('code', '')
                        parsed_item['code'] = code

                    else:
                        parsed_item['type'] = "dummy_tool"
                        parsed_item['error_msg'] = "Unkown tool name."

                    # 6. Add parsed item to result list
                    parsed_content.append(parsed_item)

                # Catch other unexpected errors
                except Exception as e:
                    parsed_item = {
                        'type': "dummy_tool",
                        'id': call_idx,
                        'raw': call,
                        'error_msg': f"Parsing arguments failed with error message: {str(e)}, please output arguments in json format string."
                    }
                    parsed_content.append(parsed_item)

            return think_content, last_content, tool, parsed_content
        else:
            # Directly return original content for other tags
            return think_content, last_content, tool, content
    else:
        assert segments[-2].strip().lstrip("</").rstrip(">").strip() in allowed_tags
        tool = segments[-2].strip().lstrip("</").rstrip(">").strip()
        content = segments[-3].strip()
        think_content = None
        last_content = segments[-1].strip()
        # Directly return original content for other tags
        return think_content, last_content, tool, content