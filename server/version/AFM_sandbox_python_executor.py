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
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from io import StringIO
import re
from unittest.mock import patch
import shutil
from typing import Any, Callable, Dict, List, Optional, Tuple
import sys
import ast

nsjail_path = os.environ.get("NSJAILPATH")

MAX_LENGTH_TRUNCATE_CONTENT = 20000
def parse_code_blobs(code_blob: str) -> str:
    """Parses the LLM's output to get any code blob inside. Will return the code directly if it's code."""
    pattern = r"```(?:py|python)?\n(.*?)\n```"
    matches = re.findall(pattern, code_blob, re.DOTALL)
    if len(matches) == 0:
        try:  # Maybe the LLM outputted a code blob directly
            ast.parse(code_blob)
            return code_blob
        except SyntaxError as e:
            if "unexpected indent" in str(e):
                raise IndentationError(f"Indentation error in code: {e}")
            raise ValueError(
                f"Your code snippet is invalid Python code. Error: {e}\n"
                f"Here is your code snippet:\n{code_blob}\n"
                "Make sure your code is valid Python and properly formatted."
            )
    return "\n\n".join(match.strip() for match in matches)



def truncate_content(content: str, max_length: int = MAX_LENGTH_TRUNCATE_CONTENT) -> str:
    if len(content) <= max_length:
        return content
    else:
        return (
            content[: max_length // 2]
            + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
            + content[-max_length // 2 :]
        )

def fix_final_answer_code(code: str) -> str:
    """
    Sometimes an LLM can try to assign a variable to final_answer, which would break the final_answer() tool.
    This function fixes this behaviour by replacing variable assignments to final_answer with final_answer_variable,
    while preserving function calls to final_answer().
    """
    # First, find if there's a direct assignment to final_answer
    # Use word boundary and negative lookbehind to ensure it's not an object attribute
    assignment_pattern = r"(?<!\.)(?<!\w)\bfinal_answer\s*="
    if "final_answer(" not in code or not re.search(assignment_pattern, code):
        # If final_answer tool is not called in this blob, then doing the replacement is hazardous because it could false the model's memory for next steps.
        # Let's not modify the code and leave the subsequent assignment error happen.
        return code

    # Pattern for replacing variable assignments
    # Looks for 'final_answer' followed by '=' with optional whitespace
    # Negative lookbehind ensures we don't match object attributes
    assignment_regex = r"(?<!\.)(?<!\w)(\bfinal_answer)(\s*=)"
    code = re.sub(assignment_regex, r"final_answer_variable\2", code)

    # Pattern for replacing variable usage but not function calls
    # Negative lookahead (?!\s*\() ensures we don't match function calls
    # Negative lookbehind (?<!\.|\w) ensures we don't match object methods or other variables
    variable_regex = r"(?<!\.)(?<!\w)(\bfinal_answer\b)(?!\s*\()"
    code = re.sub(variable_regex, "final_answer_variable", code)
    return code
def get_conda_env_paths():
    python_exec = sys.executable
    if "conda" not in python_exec:
        raise RuntimeError("Not running in a conda environment")
    
    conda_env_path = str(Path(python_exec).parent.parent)
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    
    python_path = f"{conda_env_path}/lib/{python_version}/site-packages"
    ld_path = f"{conda_env_path}/lib:/lib:/usr/lib"
    
    return python_path, ld_path

def get_env_paths():
    python_exec = sys.executable
    is_conda = (
        "conda" in python_exec or 
        "CONDA_PREFIX" in os.environ or
        "CONDA_DEFAULT_ENV" in os.environ
    )
    
    if is_conda:
        conda_env_path = os.environ.get("CONDA_PREFIX", str(Path(python_exec).parent.parent))
        python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        
        python_path = f"{conda_env_path}/lib/{python_version}/site-packages"
        ld_path = f"{conda_env_path}/lib:/usr/local/lib:/usr/lib"
        
        if "CONDA_PREFIX" in os.environ:
            ld_path = f"{os.environ['CONDA_PREFIX']}/lib:{ld_path}"
    else:
        python_path = ":".join([
            f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages",
            f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
        ])
        
        ld_path = "/usr/local/lib:/usr/lib"
        
        if python_exec.startswith("/usr/local"):
            python_path = f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages:{python_path}"
            ld_path = f"/usr/local/lib:{ld_path}"
    
    return python_path, ld_path


def run_in_nsjail(code_str, has_input=False):
    python_path, ld_path = get_env_paths()
    temp_dir = tempfile.mkdtemp(prefix="nsjail_")
    temp_work_dir = os.path.join(temp_dir, "workspace")
    os.makedirs(temp_work_dir, exist_ok=True)
    try:
        code_path = os.path.join(temp_dir, "code.py")
        with open(code_path, 'w') as f:
            f.write(code_str)
        
        nsjail_path = os.environ.get("NSJAILPATH")
        cmd = [
            nsjail_path,
            "--disable_proc",
            "--mode", "o",
            "--user", "nobody",
            "--group", "nogroup",
            "--chroot", "/",
            "--cwd", "/tmp/workspace",
            "--rlimit_as", "50000",  # 50000MB of memory space
            "--rlimit_cpu", "20",    # 20 second CPU time limit
            "--bindmount_ro", "/opt:/opt",
            "--bindmount_ro", f"{code_path}:/tmp/code.py",
            "--bindmount", f"{temp_work_dir}:/tmp/workspace",  # Readable and writable working directory
            "--bindmount_ro", "/tmp/empty:/mnt",  # Isolate /mnt
            "--env", f"PYTHONPATH={python_path}",
            "--env", f"LD_LIBRARY_PATH={ld_path}",
            "--really_quiet",
            "--",
            sys.executable,
            "/tmp/code.py"#code_str
        ]
        
        os.makedirs("/tmp/empty", exist_ok=True)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=25 
        )
        entry = {
            "success": result.returncode == 0 or result.returncode == 99,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except Exception as e:
        error_message = str(e)
        if 'timed out' in error_message:
            error_output= f"Time out. Check for infinite loops, blocked I/O (e.g., input()), or deadlocks."
        else:
            error_output =  f"{error_message}"
        entry = {
            "success": False,
            "returncode": 1,
            "stdout": "",
            "stderr": error_output
        }
    finally:
        # Clean temporary files
        # Force cleanup of temporary directories (including all subdirectories)
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f'[nsjail ]',e)
            
    return entry


def mock_input_in_code(code_blob, input_str):
    if isinstance(input_str, list):
        input_str = "\n".join(input_str)
    
    input_lines = input_str.splitlines(keepends=True)
    input_lines_without_ends = [line.rstrip('\n') for line in input_lines]
    
    def replace_input(match):
        prompt = match.group(1) if match.group(1) else ''
        return f'next(input_iterator)'

    def replace_stdin_readline(match):
        return f'sys_stdin_readline()'
    
    def replace_stdin_readlines(match):
        return f'sys_stdin_readlines()'
    
    def replace_stdin_read(match):
        return f'sys_stdin_read()'
    
    def replace_open(match):
        filename = match.group(1)
        mode = match.group(2) if match.group(2) else 'r'
        if filename in ('sys.stdin', '/dev/stdin'):
            return f'mock_open_stdin()'
        return f'open({filename}, {mode})'
    

    modified_code = re.sub(
        r'input$([^)]*)$',
        replace_input,
        code_blob
    )
    

    modified_code = re.sub(
        r'sys\.stdin\.readline$$',
        replace_stdin_readline,
        modified_code
    )
    

    modified_code = re.sub(
        r'sys\.stdin\.readlines$$',
        replace_stdin_readlines,
        modified_code
    )
    

    modified_code = re.sub(
        r'sys\.stdin\.read$$',
        replace_stdin_read,
        modified_code
    )
    

    modified_code = re.sub(
        r'open$(sys\.stdin|/dev/stdin)(?:,\s*([\'"]\w+[\'"]))?$',
        replace_open,
        modified_code
    )

    no_need_setup=f"""
from unittest.mock import mock_open, patch

input_iterator = iter({input_lines_without_ends})
def sys_stdin_readline():
    try:
        return next(input_iterator) + '\\n'
    except StopIteration:
        return ''

def sys_stdin_readlines():
    return list(input_iterator)

def sys_stdin_read():
    return {input_str!r}

def mock_open_stdin():
    return StringIO({input_str!r})
"""

    input_setup = f"""
import sys
from io import StringIO
sys_stdin_content = StringIO({input_str!r})

# 替换sys.stdin
sys.stdin = sys_stdin_content
"""
    
    final_code = input_setup + modified_code
    return final_code


def exec_nsjail(code_blob, input_str=''):
    if input_str == '':
        result = run_in_nsjail(code_blob, has_input=False)
    else:
        new_code_blob = mock_input_in_code(code_blob, input_str)
        result = run_in_nsjail(new_code_blob, has_input=True)
    
    succ = result["success"]
    if succ:
        if not result["stdout"]:
            observation = '[EXECUTED] Code exited with status 0 (no output).'
        else:
            if '__FINAL_ANSWER__' in result["stdout"]:
                observation = (
                    '[EXECUTED] Code exited with status 99.\n'
                    '[STDOUT:BEGIN]\n'
                    f'{result["stdout"].strip()}\n'
                    '[STDOUT:END]'
                )
            else:
                observation = (
                    '[EXECUTED] Code exited with status 0.\n'
                    '[STDOUT:BEGIN]\n'
                    f'{result["stdout"].strip()}\n'
                    '[STDOUT:END]'
                )
    else:
        exit_code = result.get("returncode", 1)
        if not result["stdout"] and result["stderr"]:
            observation = (
                f'[FAILED] Code exited with status {exit_code}.\n'
                '[STDERR:BEGIN]\n'
                f'{result["stderr"].strip()}\n'
                '[STDERR:END]'
            )
        elif not result["stdout"] and not result["stderr"]:
            observation = f'[FAILED] Code exited with status {exit_code} (no output).'
        elif result["stdout"] and not result["stderr"]:
            observation = (
                f'[FAILED] Code exited with status {exit_code}.\n'
                '[STDOUT:BEGIN]\n'
                f'{result["stdout"].strip()}\n'
                '[STDOUT:END]'
            )
        else:
            observation = (
                f'[FAILED] Code exited with status {exit_code} with mixed output:\n'
                '[STDOUT:BEGIN]\n'
                f'{result["stdout"].strip()}\n'
                '[STDOUT:END]\n'
                '[STDERR:BEGIN]\n'
                f'{result["stderr"].strip()}\n'
                '[STDERR:END]'
            )
    return succ, observation

def extract_output(observation):
    output = None
    is_final_answer = False
    if "[STDOUT:BEGIN]" in observation:
        std_output = re.search(r"\[STDOUT:BEGIN\]\n(.*?)\n\[STDOUT:END\]", observation, re.DOTALL)
        if std_output:
            std_output = std_output.group(1)
            if "__FINAL_ANSWER__:" in std_output:
                is_final_answer = True
                output = std_output.split("__FINAL_ANSWER__:")[-1].strip()
            else:
                output = std_output.strip()
    return output, is_final_answer


class SandboxPythonExecutor:
    
    def wrap_code_action(self, code_action):
        indent = '    '
    
        indented_code = '\n'.join(
            f"{indent}{line}" for line in code_action.split('\n')
        )

        wrapped_code = f'''
import sys
import traceback

class FinalAnswerException(Exception): 
    pass

def final_answer(value):
    raise FinalAnswerException(value)

try:
{indented_code}
except FinalAnswerException as e:
    print("__FINAL_ANSWER__:" + str(e))
    sys.exit(99)
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
'''
        return wrapped_code


    def forward(self, code_action: str, test_cases: list[Dict[str,str]]) -> Tuple[Any, str, bool]:
        wrapped_code = self.wrap_code_action(code_action)
        success_test_cases = []
        fail_test_cases = []
        just_input_cases = []
        if test_cases:
            for test_case in test_cases:
                insert_input_in_code = mock_input_in_code(wrapped_code, test_case['input'])
                succ, observation = exec_nsjail(insert_input_in_code)
                output, is_final_answer = extract_output(observation)
                if not succ:
                    output_str = (
                    f'[PASSED]: {succ}\n'
                    f'[OUTPUT]: {output}\n'
                    f'[OBSERVATIONS]: {observation}\n'
                    f'[IS_FINAL_ANSWER]: {is_final_answer}'
                    )
                    return output_str
                if is_final_answer:
                    output_str = (
                    f'[PASSED]: {succ}\n'
                    f'[OUTPUT]: {output}\n'
                    f'[OBSERVATIONS]: {observation}\n'
                    f'[IS_FINAL_ANSWER]: {is_final_answer}'
                    )
                    return output_str

                if 'output' in test_case:
                    if output != (test_case['output'].replace('\r','')[:-1].strip() if test_case['output'][-1] == '\n' else test_case['output'].replace('\r','').strip()):
                        fail_test_case = {}
                        fail_test_case['input'] = test_case['input']
                        fail_test_case['expected output'] = test_case['output']
                        fail_test_case['actual output'] = output
                        fail_test_cases.append(fail_test_case)
                    else:
                        success_test_cases.append(test_case)
                else:
                    just_input_cases.append(
                        {
                            "input": test_case["input"],
                            "output": output
                        }
                    )
            if fail_test_cases:
                observation = f"Execute some test cases incorrectly. {len(fail_test_cases)} Failed test cases: {fail_test_cases}. {len(success_test_cases)} Correct test cases: {success_test_cases}"
                if just_input_cases:
                    observation += f"\nAccepted some inputs without expected output, and the execution result is: {just_input_cases}"
                output_str = (
                    f'[PASSED]: {False}\n'
                    f'[OUTPUT]: {output}\n'
                    f'[OBSERVATIONS]: {observation}\n'
                    f'[IS_FINAL_ANSWER]: {False}'
                )
                return output_str
            elif just_input_cases:
                if success_test_cases:
                    observation = "Execute all test cases with expected output correctly."
                    observation += f"\nAccepted some inputs without expected output, and the execution result is: {just_input_cases}"
                    output_str = (
                    f'[PASSED]: {True}\n'
                    f'[OUTPUT]: {success_test_cases + just_input_cases}\n'
                    f'[OBSERVATIONS]: {observation}\n'
                    f'[IS_FINAL_ANSWER]: {False}'
                    )
                else:
                    observation = "All inputs were executed successfully."
                    output_str = (
                    f'[PASSED]: {True}\n'
                    f'[OUTPUT]: {just_input_cases}\n'
                    f'[OBSERVATIONS]: {observation}\n'
                    f'[IS_FINAL_ANSWER]: {False}'
                    )
                return output_str
            else:
                observation = "Execute all test cases correctly."
                output_str = (
                    f'[PASSED]: {True}\n'
                    f'[OUTPUT]: {output}\n'
                    f'[OBSERVATIONS]: {observation}\n'
                    f'[IS_FINAL_ANSWER]: {False}'
                )
                return output_str
        else:
            succ, observation = exec_nsjail(wrapped_code)
            output, is_final_answer = extract_output(observation)
            output_str = (
                    f'[PASSED]: {succ}\n'
                    f'[OUTPUT]: {output}\n'
                    f'[OBSERVATIONS]: {observation}\n'
                    f'[IS_FINAL_ANSWER]: {is_final_answer}'
                )
            return output_str

if __name__ == "__main__":
    executor = SandboxPythonExecutor()
    code_str = '''
T = int(input())
for _ in range(T):
    S = input().strip()
    stack = []
    balanced = True
    for c in S:
        if c == '(': 
            stack.append(c)
        elif c == ')':
            if stack:
                stack.pop()
            else:
                balanced = False
                break
    if balanced and not stack:
        print('YES')
    else:
        print('NO')
'''

    test_cases = [
            {"input": "3\n((()))\n(())()\n()(()", "output": "YES\nYES\nNO"} , 
            {"input": "3\n((()))\n(())()\n()())", "output": "YES\nYES\nNO"} , 
            {"input": "3\n((()()\n(())()\n()(()", "output": "NO\nYES\nNO\n"}
        ]
    code_str = parse_code_blobs(code_str)
    output = executor.forward(code_str, test_cases)
    print(output)