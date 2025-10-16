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
import asyncio
import concurrent.futures
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple, Union

import aiohttp
import requests
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

# Add servers/ to PYTHONPATH
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AFM_sandbox_python_executor import (
    SandboxPythonExecutor,
    parse_code_blobs,
    truncate_content,
    fix_final_answer_code
)

from dotenv import load_dotenv
load_dotenv(override=True)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

CODE_EXECUTION_TIMEOUT = 30


class CodeExecuteRequest(BaseModel):
    code_str_list: List[str] = Field(..., description="List of code str to execute")
    parameter_list: List[str] = Field(..., description="List of code parameters when executing the code")

class CodeExecuteResponse(BaseModel):
    success: bool
    obs: List = None
    error_message: Optional[str] = None
    processing_time: float

class CodeExecuteServer:
    def __init__(self):
        logger.info("Initializing CodeExecuteServer")
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)

        self.code_execution_timeout = CODE_EXECUTION_TIMEOUT
        logger.info(f"CodeExecuteServer initialized with code_execution_timeout={self.code_execution_timeout}s")

    def extract_info(self, observation):
        patterns = {
        'PASSED': r'\[PASSED\]:\s*(\S+)',
        'OUTPUT': r'\[OUTPUT\]:\s*([\s\S]*?)(?=\n\[|$)',
        'OBSERVATIONS': r'\[OBSERVATIONS\]:\s*([\s\S]*?)(?=\n\[|$)',
        'STDOUT': r'\[STDOUT:BEGIN\]\n([\s\S]*?)\n\[STDOUT:END\]',
        'STDERR': r'\[STDERR:BEGIN\]\n([\s\S]*?)\n\[STDERR:END\]',
        'IS_FINAL_ANSWER': r'\[IS_FINAL_ANSWER\]:\s*(\S+)'
        }

        result = {'PASSED':'', 'OUTPUT':'', 'OBSERVATIONS':'', 'STDOUT':'', 'STDERR':'', 'IS_FINAL_ANSWER':''}

        for field, pattern in patterns.items():
            match = re.search(pattern, observation)
            if match:
                result[field] = match.group(1).strip()
        observation = result['OBSERVATIONS']
        truncated_output = truncate_content(str(result['OUTPUT']))
    

        if result['PASSED'] == 'False' and result['STDERR']:
            observation += "\nError message:\n" + result['STDERR']
        return result['PASSED'] == 'True', truncated_output, observation, result['IS_FINAL_ANSWER'] == 'True'


    async def execute_code(self, request: CodeExecuteRequest) -> CodeExecuteResponse:
        start_time = time.time()
        python_executor = SandboxPythonExecutor()
        code_execution_results = []
        try:
            logger.info("--------- Start processing the code_exec request ---------")
            logger.info(f"Processing {len(request.code_str_list)} code snippets: \n{request.code_str_list}\n")

            loop = asyncio.get_running_loop()

            for code_str in request.code_str_list:
                code_str = fix_final_answer_code(parse_code_blobs(code_str))
                code_output_bundle = await loop.run_in_executor(
                    self.thread_pool,
                    python_executor.forward, 
                    code_str,
                    []
                )
                exec_status, output, observation, is_final_answer = self.extract_info(code_output_bundle)
                logger.info(f"--------------- Code Execution: --------------- \n{code_output_bundle}")
                logger.info("------------------------------------------------")
                code_execution_results.append((exec_status, output, observation))
            



            processing_time = time.time() - start_time
            logger.info(f"code exec done, cost time: {processing_time:.2f} seconds.")
            logger.info("--------- Request processed successfully ---------")
            return CodeExecuteResponse(
                success=True,
                obs=code_execution_results,
                processing_time=processing_time
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"code exec error: {str(e)}", exc_info=True)
            logger.error("--------- Request processing failed ---------")
            return CodeExecuteResponse(
                success=False,
                obs="",
                error_message=f"server error: {str(e)}",
                processing_time=processing_time
            )

code_exec_server = CodeExecuteServer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("CodeExec Server startup")
    yield
    logger.info("CodeExec Server closed")

app = FastAPI(
    title="CodeExec tool server",
    description="A code_exec tool service based on FastAPI, supporting high concurrency",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/code_exec", response_model=CodeExecuteResponse)
async def code_exec_endpoint(request: CodeExecuteRequest):
    logger.info(f"Received code_exec request from client")
    try:
        result = await asyncio.wait_for(
            code_exec_server.execute_code(request),
            timeout=CODE_EXECUTION_TIMEOUT
        )
        logger.info(f"Request completed successfully, success={result.success}")
        return result
    except asyncio.TimeoutError:
        logger.error(f"Request timeout after {CODE_EXECUTION_TIMEOUT}s")
        raise HTTPException(status_code=504, detail=f"request timeout: {CODE_EXECUTION_TIMEOUT} seconds")
    except Exception as e:
        logger.error(f"Interface processing exception: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/")
async def root():
    return {
        "message": "CodeExec tool server",
        "version": "1.0.0",
        "endpoints": {
            "code_exec": "/code_exec",
            "health": "/health",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    host = os.getenv("SERVER_HOST", None)
    port = os.getenv("CODE_EXEC_PORT", None)
    if port == None:
        raise NotImplementedError("[ERROR] CODE_EXEC_PORT NOT SET!")
    port = int(port)

    # start
    logger.info(f"Configuring server with host={host}, port={port}, workers=10")
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        workers=10,
        reload=False,
        access_log=True,
        log_level="info"
    ) 
    server = uvicorn.Server(config)
    try:
        logger.info(f"Start the CODE_EXECUTION server... http://{host}:{port}")
        server.run()
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        raise
