import asyncio
import concurrent.futures
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple, Union
import random

import aiohttp
import requests
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

CRAWL_PAGE_TIMEOUT = 500

class CrawlPageRequest(BaseModel):
    urls: List[str] = Field(..., description="List of URLs to crawl")
    think_content: str = Field(..., description="Thought content, used to guide the summary.")
    web_search_query: str = Field(..., description="Web search query")
    summary_type: Optional[str] = Field("page", description="Summary type")
    summary_prompt_type: Optional[str] = Field("webthinker_with_goal", description="Abstract Prompt Template Type")

    # API Configuration
    api_url: Optional[str] = Field(..., description="API url")
    api_key: Optional[str] = Field(..., description="API key")
    model: Optional[str] = Field(..., description="model name")

    # legacy
    task: str = Field(None, description="Task description")
    messages: Optional[List[Dict]] = Field(None, description="Message history (optional, for future use)")

class CrawlPageResponse(BaseModel):
    success: bool
    obs: str
    error_message: Optional[str] = None
    processing_time: float

class CrawlPageServer:
    def __init__(self):
        logger.info("Initializing CrawlPageServer")
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.jina_timeout = 30
        self.summary_timeout = 300
        self.max_retries = 5
        self.jina_token_budget = 200000
        self.api_key_list = os.environ.get("JINA_API_KEY","").split("|")
        logger.info(f"API_KEY_LIST: {self.api_key_list}")
        assert self.api_key_list != [], "No api key configured."
        logger.info(f"CrawlPageServer initialized with jina_timeout={self.jina_timeout}s, summary_timeout={self.summary_timeout}s, token_budget={self.jina_token_budget}")
    
    def _select_api_key_random(self):
        return random.randint(0, len(self.api_key_list) - 1)

    def _select_api_key_with_round_robin(self, api_key_index):
        
        return (api_key_index + 1) % len(self.api_key_list)
    
    async def _fetch_with_api(self, session: aiohttp.ClientSession, url: str, base_delay: float = 1.0, max_delay: float = 16.0) -> Tuple[str, str]:
        """
        Polling and crawling a single URL using the API.
        The timeout for each request is timeout seconds.
        If successful, return (content, url) immediately; if all attempts fail, return (error_msg, url).
        """

        api_key_index = self._select_api_key_random()
        logger.info(f"Choosing api_key: {self.api_key_list[api_key_index]}")
        try_times = 0
        try:
            try_times += 1
            results = await self._fetch_with_retry(session, url, base_delay, max_delay, self.api_key_list[api_key_index])
            if isinstance(results,tuple) and "[Page content not accessible" in results[0]:
                raise Exception("Unsuccessful crawl")
            logger.info(f"Successfully crawled page with api_key ending with {self.api_key_list[api_key_index][-5:]}")
        except Exception as first_error:
            logger.warning(f"API Key with ending {self.api_key_list[api_key_index][-5:]} failed, trying alternatives")

            if try_times == len(self.api_key_list):
                # There are no other available API Keys.
                logger.warning(f"API Key with ending {self.api_key_list[api_key_index][-5:]} failed, no alternatives left")
                raise first_error
            
            # Try the remaining API Keys
            last_error = first_error
            while try_times < len(self.api_key_list):
                try:
                    try_times += 1
                    api_key_index = self._select_api_key_with_round_robin(api_key_index)
                    logger.info(f"Choosing api_key: {self.api_key_list[api_key_index]}")
                    results = await self._fetch_with_retry(session, url, base_delay, max_delay, self.api_key_list[api_key_index])
                    if isinstance(results,tuple) and "[Page content not accessible" in results[0]:
                        raise Exception("Unsuccessful crawl")
                    logger.info(f"Successfully crawled page with api_key ending with {self.api_key_list[api_key_index][-5:]}")
                    break
                except Exception as e:
                    logger.warning(f"API Key with ending {self.api_key_list[api_key_index][-5:]} failed, trying alternatives")
                    last_error = e
            else:
                # All API Keys have failed
                raise last_error
        return results

    async def _fetch_with_retry(self, session: aiohttp.ClientSession, url: str, base_delay: float = 1.0, max_delay: float = 16.0, apikey: str = '') -> Tuple[str, str]:
        """
        Crawl a single URL at most self.max_retries times.
        Each request times out after timeout seconds.
        If successful, immediately return (content, url); if all attempts fail, return (error_msg, url).
        """
        assert apikey != "", "No api key when fetching."
        attempt = 0
        last_exc = None

        while attempt < self.max_retries:
            attempt += 1
            try:
                logger.info(f"[Attempt {attempt}/{self.max_retries}] {url}")
                timeout = aiohttp.ClientTimeout(total=self.jina_timeout)
                jina_url = f"https://r.jina.ai/{url}"
                headers = {
                    'Authorization': f'Bearer {apikey}',
                    'X-Engine': 'browser',
                    'X-Return-Format': 'text',
                    "X-Remove-Selector": "header, .class, #id",
                    'X-Timeout': str(self.jina_timeout),
                    "X-Retain-Images": "none",
                    'X-Token-Budget': "200000"
                }

                async with session.get(jina_url, headers=headers, timeout=timeout) as resp:
                    resp.raise_for_status()
                    content = await resp.text()
                    return content, url
            except asyncio.TimeoutError:
                last_exc = f"Timeout after {self.jina_timeout}s"
                logger.warning(f"[Attempt {attempt}] Timeout for {url}")
            except Exception as e:
                last_exc = str(e)
                logger.warning(f"[Attempt {attempt}] Error for {url}: {e}")

            if attempt < self.max_retries:
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                await asyncio.sleep(delay)

        return f"[Page content not accessible: {last_exc}]", url

    async def read_page_async(self, session: aiohttp.ClientSession, url: str) -> Tuple[str, str]:
        return await self._fetch_with_api(session, url)


    def validate_urls(self, urls: List[str]) -> List[str]:
        """Validate HTTP/HTTPS URLs."""
        processed_urls = []
        for url in urls:
            url = url.strip()
            if not url:
                continue
            if url.startswith(('http://', 'https://')):
                processed_urls.append(url)
            else:
                logger.warning(f"Invalid URL format (must start with http:// or https://): {url}")        
        return processed_urls

    def get_click_intent_instruction(self, prev_reasoning: str) -> str:
        return f"""Based on the previous thoughts below, provide the detailed intent of the latest click action.
    Previous thoughts: {prev_reasoning}
    Please provide the current click intent."""

    def get_summary_prompt_new(self, query: str, content: str) -> str:
        content = content[:60000]
        """Generate prompt for content summarization."""
        return (
            f"Task: Extract all content from the web page that matches the search query.\n"
            f"Search Query: {query}\n\n"
            f"Web Page Content:\n{content}\n\n"
            "Instructions:\n"
            "- Summarize all relevant content for the query (text, tables, lists) into concise points\n"
            "- If no relevant information exists, please straightly output 'No relevant information'\n"
            "- Keep the summary under 500 words"
        )

    def get_summary_prompt(self, web_search_query: str, think_content: str, page_contents: str, summary_prompt_type: str = "webthinker", click_intent: str = "") -> str:
        page_contents = page_contents[:60000]
        if summary_prompt_type == "webthinker":
            prompt = f"""
        Target: Extract all content from a web page that matches a specific web search query and search query, ensuring completeness and relevance. (No response/analysis required.)
        
        web search query: 
        {web_search_query}

        Clues and ideas: 
        {think_content}
        
        Searched Web Page: 
        {page_contents}

        Important Notes:
        - Summarize all content (text, tables, lists, code blocks) into concise points that directly address query and clues and ideas.
        - Preserve and list all relevant links ([text](url)) from the web page.
        - Summarize in three points: web search query-related information, clues and ideas-related information, and relevant links with descriptions.
        - If no relevant information exists, Just output "No relevant information"
        """
        elif summary_prompt_type == "webdancer":
            prompt = f"""Please process the following webpage content and user goal to extract relevant information:

        ## **Webpage Content** 
        {page_contents}

        ## **User Goal**
        {think_content}

        ## **Task Guidelines**
        1. **Content Scanning**: Locate the **specific sections/data** directly related to the user's goal within the webpage content.
        2. **Key Extraction**: Identify and extract the **most relevant information** from the content, you never miss any important information
        3. **Summary Output**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.
        4. **relevant links**: Preserve and list all relevant links ([text](url)) from the web page.

        **Final Output**:
        1."rational": "string"
        2."evidence": "string"
        3."summary": "string"
        4."relative urls": ([text](url), [text](url) ...)
        """
        elif summary_prompt_type == "webthinker_with_goal":
            prompt = f"""
        Target: Extract all content from a web page that matches a specific web search query and search query, ensuring completeness and relevance. (No response/analysis required.)
        
        web search query: 
        {web_search_query}

        Clues and ideas: 
        {click_intent}
        
        Searched Web Page: 
        {page_contents}

        Important Notes:
        - Summarize all content (text, tables, lists, code blocks) into concise points that directly address query and clues and ideas.
        - Preserve and list all relevant links ([text](url)) from the web page.
        - Summarize in three points: web search query-related information, clues and ideas-related information, and relevant links with descriptions.
        - If no relevant information exists, Just output "No relevant information"
        """
        elif summary_prompt_type == "webdancer_with_goal":
            prompt = f"""Please process the following webpage content and user goal to extract relevant information:

        ## **Webpage Content**
        {page_contents}

        ## **User Goal**
        {click_intent}

        ## **Task Guidelines**
        1. **Content Scanning**: Locate the **specific sections/data** directly related to the user's goal within the webpage content.
        2. **Key Extraction**: Identify and extract the **most relevant information** from the content, you never miss any important information
        3. **Summary Output**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.
        4. **relevant links**: Preserve and list all relevant links ([text](url)) from the web page.

        **Final Output**:
        1."rational": "string"
        2."evidence": "string"
        3."summary": "string"
        4."relative urls": ([text](url), [text](url) ...)
        """
        else:
            raise ValueError(f"Invalid summary prompt type: {summary_prompt_type}")
        return prompt

    async def call_ai_api_async(self, system_prompt: str, user_prompt: str, api_url: str, api_key: str, model: str, max_retries: int = 5, base_delay: float = 5) -> str:
        logger.info(f"Calling AI API with model: {model}, API URL: {api_url}, max_retries: {max_retries}")
        attempt = 0
        last_error = None

        while attempt < max_retries:
            attempt += 1
            try:
                logger.info(f"[Attempt {attempt}/{max_retries}] Calling AI API...")
                client = AsyncOpenAI(base_url=api_url, api_key=api_key)
                completion = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    extra_headers={
                        'X-DashScope-DataInspection':'{"input":"disable","output":"disable"}'  
                    },
                    stream=False,
                    timeout=self.summary_timeout
                )
                content = completion.choices[0].message.content
                logger.info(f"AI API response received, length: {len(content)} chars")
                return content
            except Exception as e:
                last_error = str(e)
                logger.warning(f"[Attempt {attempt}] AI API Call failed: {last_error}, API_KEY: {api_key}, API_URL: {api_url}")
                if attempt < max_retries:
                    delay = base_delay * (2 ** (attempt - 1)) # 60s -> 120s -> 240s -> 480s
                    logger.info(f"Retry after waiting {delay:.2f}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("All retry attempts have been exhausted, and the AI API call failed.")

        return f"AI processing failed (after retrying {max_retries} times): {last_error}"
        
    async def summarize_content(self, content: str, request: CrawlPageRequest) -> str:
        """Helper function to summarize content"""
        logger.info(f"Summarizing content of length: {len(content)} chars")
        detailed_prompt = self.get_summary_prompt_new(
            request.web_search_query, content
        )
        return await self.call_ai_api_async(
            "You are a summary agent robot.", detailed_prompt,
            request.api_url, request.api_key, request.model
        )
    
    async def process_crawl_page(self, request: CrawlPageRequest) -> CrawlPageResponse:
        start_time = time.time()
        try:
            logger.info("--------- Start processing the crawl_page request ---------")
            logger.info(f"Processing {len(request.urls)} URLs: {request.urls}")
            logger.info(f"web_search_query='{request.web_search_query}'")
            logger.info(f"Summary type: {request.summary_type}")
            
            # Validate and clean up the URL list
            urls = self.validate_urls(request.urls)
            if not urls:
                logger.warning("No valid URLs found after validation")
                return CrawlPageResponse(
                    success=False,
                    obs="",
                    error_message="No valid URLs found",
                    processing_time=time.time() - start_time
                )
            
            logger.info(f"Start processing {len(urls)} URLs: {urls}")
            
            # Asynchronously fetching page content
            page_contents = ""
            logger.info("Creating aiohttp session for page fetching")
            async with aiohttp.ClientSession() as session:
                tasks = [self.read_page_async(session, url) for url in urls]
                logger.info(f"Fetching {len(tasks)} pages concurrently")
                page_results = await asyncio.gather(*tasks, return_exceptions=True)

                processed_results = []
                for i, result in enumerate(page_results):
                    if isinstance(result, Exception):
                        logger.error(f"Unhandled exception for URL {urls[i]}: {result}")
                        processed_results.append((f"[Page content not accessible: {result}]", urls[i]))
                    else:
                        processed_results.append(result)
                page_results = processed_results

            ##### end Jina read page #####
            logger.info(f"Page fetching completed after {time.time() - start_time} seconds")
            
            ##### begin page summary #####
            summary_type = request.summary_type
            logger.info(f"Using summary type: {summary_type}")
            if summary_type == "none":
                # No summarization, just concatenate all content
                logger.info("No summarization requested, concatenating raw content")
                summary_result = "\n\n".join(f"Page {i+1} [{result[1]}]: {result[0]}" for i, result in enumerate(page_results))
                logger.info(f"Combined content length: {len(summary_result)} chars")

            elif summary_type == "once":
                # Single summarization of all content
                logger.info("Using 'once' strategy - single summarization of all content")
                page_contents = "\n\n".join(f"Page {i+1} [{result[1]}]: {result[0]}" for i, result in enumerate(page_results))
                logger.info(f"Combined content for summarization: {len(page_contents)} chars")
                summary_result = await self.summarize_content(page_contents, request)

            elif summary_type == "page":
                # Page-by-page summarization
                logger.info(f"Using page summary strategy")
                
                # Process all pages concurrently
                page_tasks = []
                page_indices = []  # Record which pages need to be summarized
                for i, (content, url) in enumerate(page_results):
                    logger.info(f"Creating task for page {i+1}/{len(page_results)}, URL: {url}, content length: {len(content) / 1000:.2f}k characters")
                    if content.startswith("[Page content not accessible:"):
                        # For pages that cannot be accessed, do not create tasks; handle them directly later.
                        page_indices.append((i, False))  # False indicates that a summary is not needed
                    else:
                        # Create a summary task
                        task = self.summarize_content(content, request)
                        page_tasks.append(task)
                        page_indices.append((i, True))  
                
                # Wait for all page summaries to be completed
                if page_tasks:
                    logger.info(f"Processing {len(page_tasks)} pages concurrently")
                    page_results_summary = await asyncio.gather(*page_tasks, return_exceptions=True)
                else:
                    page_results_summary = []
                
                # Processing result
                page_summaries = []
                summary_idx = 0
                for i, needs_summary in page_indices:
                    content, url = page_results[i]
                    if not needs_summary:
                        # Unreachable page, directly use the original content
                        page_summaries.append(f"Page {i+1} [{url}]: {content}")
                    else:
                        # Get the summary result
                        result = page_results_summary[summary_idx]
                        summary_idx += 1
                        if isinstance(result, Exception):
                            logger.error(f"Error processing page {i+1} [{url}]: {str(result)}")
                            page_summaries.append(f"Page {i+1} [{url}] Summary:\n[Error: {str(result)}]")
                        else:
                            page_summaries.append(f"Page {i+1} [{url}] Summary:\n{result}")

                summary_result = "\n\n".join(page_summaries)
            else:
                logger.error(f"Invalid summary_type: {summary_type}")
                raise ValueError(f"Invalid summary_type: {summary_type}, only support 'none', 'once', 'page'")
            
            processing_time = time.time() - start_time
            logger.info(f"crawl page done, cost time: {processing_time:.2f} seconds, result length: {len(summary_result)} chars")
            logger.info("--------- Request processed successfully ---------")
            
            return CrawlPageResponse(
                success=True,
                obs=summary_result,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"crawl page error: {str(e)}", exc_info=True)
            logger.error("--------- Request processing failed ---------")
            return CrawlPageResponse(
                success=False,
                obs="",
                error_message=f"crawl page error: {str(e)}",
                processing_time=processing_time
            )


crawl_server = CrawlPageServer()

# Create a FastAPI application
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("CrawlPage Server startup")
    yield
    logger.info("CrawlPage Server closes")

app = FastAPI(
    title="CrawlPage tool server",
    description="A crawl_page tool service based on FastAPI, supporting high concurrency and fault tolerance",
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


@app.post("/crawl_page", response_model=CrawlPageResponse)
async def crawl_page_endpoint(request: CrawlPageRequest):
    logger.info(f"Received crawl_page request from client")
    try:
        result = await asyncio.wait_for(
            crawl_server.process_crawl_page(request),
            timeout=CRAWL_PAGE_TIMEOUT
        )
        logger.info(f"Request completed successfully, success={result.success}")
        return result
    except asyncio.TimeoutError:
        logger.error(f"Request timeout after {CRAWL_PAGE_TIMEOUT}s")
        raise HTTPException(status_code=504, detail=f"request timeout: {CRAWL_PAGE_TIMEOUT} seconds")
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
        "message": "CrawlPage tool server",
        "version": "1.0.0",
        "endpoints": {
            "crawl_page": "/crawl_page",
            "health": "/health",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    host = os.getenv("SERVER_HOST", None)
    port = os.getenv("CRAWL_PAGE_PORT", None)
    if port == None:
        raise NotImplementedError("[ERROR] CRAWL_PAGE_PORT NOT SET!")
    port = int(port)

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
        logger.info(f"Start CrawlPage Server... http://{host}:{port}")
        server.run()
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        raise