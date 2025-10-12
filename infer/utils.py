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


def truncate_special_tokens(text: str, max_tokens: int, tokenizer: str) -> str:
    """
    Divide the text according to special tokens, count the number of tokens, and retain the content from the end based on the maximum number of tokens.
    Only retain the <plan>, <|FunctionExecute|> tags and the one preceding <think> tag, and delete other tags.
    
    Parameters:
    text (str): Text containing special tokens
    max_tokens (int): Maximum number of tokens to retain
    model (str): Model name used for tokenization, default is "gpt-4-1106-preview"
    
    Returns:
    str: Truncated text
    """
    
    # Define the regular expression pattern for special tokens
    token_pattern = r'<[^>]+>|</[^>]+>'
    
    # Split the text according to special tokens
    segments = re.split(f'({token_pattern})', text)
    segments = [seg for seg in segments if seg] 
    
    if not segments:
        return ""
    
    # Calculate the number of tokens in each segment
    segment_tokens = []
    for segment in segments:
        tokens = tokenizer.encode(segment)
        segment_tokens.append((segment, len(tokens)))
    
    # Find the positions of all plan and reflection tags
    keep_tags = ['plan', 'reflection']
    keep_tag_indices = []
    
    for i, (segment, _) in enumerate(segment_tokens):
        tag_match = re.match(r'<(/?)([^>]+)>', segment)
        if tag_match:
            is_closing = tag_match.group(1) == '/'
            tag_name = tag_match.group(2)
            if tag_name in keep_tags:
                keep_tag_indices.append((i, tag_name, is_closing))
    
    # Collect the indexes of special tag fragments that need to be retained
    special_indices = set()
    
    # Process each special tag to ensure that the tag pairs are complete.
    for i, tag_name, is_closing in keep_tag_indices:
        if is_closing:
            # Closing tag, find the corresponding opening tag
            start_idx = -1
            depth = 0
            for j in range(i, -1, -1):
                seg = segment_tokens[j][0]
                m = re.match(r'<(/?)([^>]+)>', seg)
                if m:
                    current_tag = m.group(2)
                    if current_tag == tag_name:
                        if m.group(1) == '/':
                            depth += 1
                        else:
                            depth -= 1
                            if depth == 0:
                                start_idx = j
                                break
            if start_idx != -1:
                for idx in range(start_idx, i + 1):
                    special_indices.add(idx)
        else:
            # Start tag, find the corresponding end tag
            end_idx = -1
            depth = 0
            for j in range(i, len(segments)):
                seg = segment_tokens[j][0]
                m = re.match(r'<(/?)([^>]+)>', seg)
                if m:
                    current_tag = m.group(2)
                    if current_tag == tag_name:
                        if m.group(1) == '/':
                            depth -= 1
                            if depth == 0:
                                end_idx = j
                                break
                        else:
                            depth += 1
            if end_idx != -1:
                for idx in range(i, end_idx + 1):
                    special_indices.add(idx)
    
    # Count the number of tokens in the special tag section
    special_segments = [segment_tokens[i] for i in sorted(special_indices)]
    special_tokens = sum(token_count for _, token_count in special_segments)
    
    # If the part with special tags has exceeded the token limit, truncate it and return.
    if special_tokens > max_tokens:
        result = []
        current_tokens = 0
        
        for segment, token_count in reversed(special_segments):
            if current_tokens + token_count <= max_tokens:
                result.insert(0, segment)
                current_tokens += token_count
            else:
                if re.match(r'</[^>]+>', segment):
                    tag_name = re.search(r'</([^>]+)>', segment).group(1)
                    start_tag_found = False
                    for s, _ in reversed(result):
                        if re.match(rf'<{tag_name}>', s):
                            start_tag_found = True
                            break
                    if not start_tag_found:
                        for s, tc in reversed(special_segments):
                            if re.match(rf'<{tag_name}>', s):
                                if current_tokens + tc <= max_tokens:
                                    result.insert(0, s)
                                    current_tokens += tc
                                break
        return ''.join(result) if result else segments[0]
    
    # There are remaining tokens; add other content from the end to the beginning.
    remaining_tokens = max_tokens - special_tokens
    additional_segments = []
    
    # Traverse all fragments from back to front
    for i in range(len(segments) - 1, -1, -1):
        if i in special_indices:
            continue  
        
        segment, token_count = segment_tokens[i]
        
        # If adding the current fragment would exceed the limit, try adding part of it or skip it.
        if remaining_tokens <= 0:
            break
        
        if token_count <= remaining_tokens:
            additional_segments.insert(0, segment)
            remaining_tokens -= token_count
        else:
            # Try to partially add text content
            if not re.match(r'<[^>]+>|</[^>]+>', segment): 
                tokens = tokenizer.encode(segment)
                partial_tokens = tokens[:remaining_tokens]
                partial_text = tokenizer.decode(partial_tokens)
                if partial_text:
                    additional_segments.insert(0, partial_text)
                    remaining_tokens = 0
    
    # Combination result: special tag part + other content added from back to front
    return ''.join([seg for seg, _ in special_segments] + additional_segments)

def extract_specific_tag_parallel_toolcalldict(text, allowed_tags=None):
    if allowed_tags is None:
        allowed_tags = ['answer', 'plan', 'summary', 'classification', 'reasoning','tool_call']

    tag_stack = deque()
    tag_pairs = []
    # Improved regular expression to ensure matching all allowed tags
    split_pattern = re.compile(r'(<\/?(?:{})>)'.format('|'.join(allowed_tags)))
    segments = split_pattern.split(text)
    segments = [s for s in segments if s.strip()]

    # Extract information from the last tag and process it
    if segments[-1].strip().lstrip("</").rstrip(">").strip() in allowed_tags:
        tool = segments[-1].strip().lstrip("</").rstrip(">").strip()
        content = segments[-2].strip()
        assert tool in segments[-3]
        think_content = segments[-4].strip() if len(segments) > 3 else None
        last_content = None

        # Process tool_calls tag, supporting multiple tool calls
        if tool == "tool_call":
            parsed_content = []

            tool_calls = content.split("\n{'id':")
            new_tool_calls = [tool_calls[0]]
            for tool_call in tool_calls[1:]:
                tool_call = "{'id':"+tool_call
                new_tool_calls.append(tool_call)

            tool_calls = new_tool_calls

            # Clean whitespace characters for each call
            tool_calls = [call.strip() for call in tool_calls if call.strip()]

            for call in tool_calls:
                try:
                    # Parse JSON format tool call
                    call_dict = ast.literal_eval(call)

                    # Extract key information
                    tool_type = call_dict.get('name')
                    if not tool_type:
                        continue

                    # Extract parameters
                    arguments = call_dict.get('arguments', {})

                    # Organize return data based on tool type
                    parsed_item = {
                        'type': tool_type,
                        'id': call_dict.get('id'),
                        'raw': call,
                    }

                    # Add specific type parameters
                    if tool_type == 'web_search':
                        parsed_item['query'] = arguments.get('query', '')
                    elif tool_type == 'crawl_page':
                        parsed_item['url'] = arguments.get('url', '')
                        parsed_item['query'] = arguments.get('query', '')
                    elif tool_type == 'code_execute':
                        # Preserve line breaks in code
                        parsed_item['code'] = arguments.get('code', '')

                    parsed_content.append(parsed_item)

                except (SyntaxError, ValueError, KeyError) as e:
                    # Handle parsing errors
                    continue

            return think_content, last_content, tool, parsed_content
        else:
            # For other tags, return original content directly
            return think_content, last_content, tool, content
    else:
        assert segments[-2].strip().lstrip("</").rstrip(">").strip() in allowed_tags
        tool = segments[-2].strip().lstrip("</").rstrip(">").strip()
        content = segments[-3].strip()
        think_content = None
        last_content = segments[-1].strip()
        # For other tags, return original content directly
        return think_content, last_content, tool, content
