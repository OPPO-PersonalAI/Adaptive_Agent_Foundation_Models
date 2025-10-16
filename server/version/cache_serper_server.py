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
import atexit
import json
import logging
import os
import threading
import time
from typing import Dict, Optional, Any
import random

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request

from dotenv import load_dotenv
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SERPER_TIMEOUT = 60
CRAWL_PAGE_TIMEOUT = 500

# --- Cache Backend --- #
class CacheBackend:
    def get(self, key: str) -> Optional[str]:
        raise NotImplementedError

    def set(self, key: str, value: str, ttl: Optional[int] = None):
        raise NotImplementedError


class InMemoryCache(CacheBackend):
    def __init__(self, file_path: str):
        self._cache: Dict[str, str] = {}
        self._lock = threading.Lock()
        self.file_path = file_path
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        # Open the file in append mode, creating it if it doesn't exist.
        self._file_handler = open(self.file_path, "a")
        self._load_from_file()
        atexit.register(self._close_file)

    def _load_from_file(self):
        try:
            # We need to read the file first, as it's opened in append mode.
            with open(self.file_path, "r") as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    try:
                        # Each line is a JSON object with one key-value pair
                        entry = json.loads(line)
                        key, value = list(entry.items())[0]
                        self._cache[key] = value
                    except (json.JSONDecodeError, IndexError) as e:
                        logger.warning(
                            f"Skipping malformed line {i+1} in {self.file_path}: {e}"
                        )
            logger.info(
                f"Cache loaded from {self.file_path}, containing {len(self._cache)} items."
            )
        except FileNotFoundError:
            logger.info(
                f"Cache file {self.file_path} not found. A new one will be created."
            )
        except Exception as e:
            logger.warning(f"Could not load cache from {self.file_path}: {e}")

    def _close_file(self):
        """Closes the file handler on exit."""
        if self._file_handler:
            self._file_handler.close()
            logger.info("Cache file handler closed.")

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value: str, ttl: Optional[int] = None):
        """Sets a value in the cache and appends it to the JSONL file."""
        with self._lock:
            if key not in self._cache:
                self._cache[key] = value
                try:
                    json.dump({key: value}, self._file_handler)
                    self._file_handler.write("\n")
                    self._file_handler.flush()  # Ensure it's written to disk
                except IOError as e:
                    logger.error(f"Could not write to cache file {self.file_path}: {e}")


class SerperProxyServer:
    def __init__(self, cache_backend: CacheBackend):
        self.cache = cache_backend

        server_host = os.environ.get("SERVER_HOST")
        crawl_page_port = os.environ.get("CRAWL_PAGE_PORT")
        self.crawl_page_endpoint = f"http:{server_host}:{crawl_page_port}/crawl_page"
        self.api_key_list = os.environ.get("WEB_SEARCH_SERPER_API_KEY","").split('|')
        assert self.api_key_list != [] , "No api keys configured."
        self.serpapi_base_url = os.environ.get("SERPAPI_BASE_URL","")
        assert self.serpapi_base_url != "", "No serpapi_base_url configured."

        # For hit rate logging
        self.total_requests = 0
        self.cache_hits = 0
        self._stats_lock = threading.Lock()
        self.history = []

        # default using first api key
        self._last_api_key_index = 0

    def _generate_cache_key(self, payload: dict) -> str:
        sorted_payload = json.dumps(payload, sort_keys=True)
        return f"serper:{sorted_payload}"

    def _log_hit_rate(self, *, hit: bool):
        """Logs the cache hit rate."""
        with self._stats_lock:
            self.total_requests += 1
            if hit:
                self.cache_hits += 1

            hit_rate = (
                (self.cache_hits / self.total_requests) * 100
                if self.total_requests > 0
                else 0
            )
            status = "HIT" if hit else "MISS"
            logger.info(
                f"Cache {status}. Rate: {hit_rate:.2f}% ({self.cache_hits}/{self.total_requests})"
            )

    def _get_header_case_insensitive(self, headers: dict, header_name: str) -> str:
        header_name_lower = header_name.lower()
        for key, value in headers.items():
            if key.lower() == header_name_lower:
                return value
        return None

    def _format_results_to_string(self, serper_json: Dict[str, Any], query: str) -> str:
        """Formats the Serper JSON result into a structured string."""
        if "organic" not in serper_json or not serper_json["organic"]:
            return f"No results found for query: '{query}'. Use a less specific query."

        web_snippets = []
        for idx, page in enumerate(serper_json["organic"], 1):
            title = page.get("title", "No Title")
            link = page.get("link", "#")
            date_published = f"\nDate published: {page['date']}" if "date" in page else ""
            source = f"\nSource: {page.get('source', '')}" if "source" in page else ""
            snippet = f"\n{page.get('snippet', '')}".replace("Your browser can't play this video.", "")

            formatted_entry = (
                f"{idx}. [{title}]({link})"
                f"{date_published}{source}"
                f"\n{link}{snippet}"
            )
            web_snippets.append(formatted_entry.strip())
        
        num_results = len(web_snippets)
        return (
            f"A web search for '{query}' found {num_results} results:\n\n## Web Results\n"
            + "\n\n".join(web_snippets)
        )

    def _to_contents_multiqueries(self, search_results: dict):
        """
        Convert the search results dictionary to a string

        Args:
            search_results: Search results in dictionary format, where the key is the query string and the value is a list of results.
            The value can be a string (error message) or a list of dictionaries (normal search results)
        Returns:
            String
        """
        all_contents = []
        total_results = 0

        for query, snippets in search_results.items():
            if isinstance(snippets, str):
                all_contents.append(f"## Query: '{query}'\n{snippets}")
                continue

            # Process normal search results
            web_snippets = []
            idx = 1
            for search_info in snippets:
                if isinstance(search_info, dict):
                    # Ensure that the dictionary contains the necessary keys
                    title = search_info.get('title', 'No title')
                    link = search_info.get('link', '#')
                    date = search_info.get('date', '')
                    source = search_info.get('source', '')
                    snippet = search_info.get('snippet', 'No snippet available')

                    redacted_version = (
                        f"{idx}. [{title}]({link})"
                        f"{date}{source}\n"
                        f"{self._pre_visit(link)}{snippet}"
                    ).replace("Your browser can't play this video.", "")

                    web_snippets.append(redacted_version)
                    idx += 1

            query_content = (
                    f"## Query: '{query}'\n"
                    f"Found {len(web_snippets)} results:\n\n"
                    + "\n\n".join(web_snippets)
            )
            all_contents.append(query_content)
            total_results += len(web_snippets)

        # Add total result count statistics
        if total_results > 0:
            summary = f"# Search Summary\nTotal results: {total_results}\n\n"
            return summary + "\n\n".join(all_contents)
        return "\n\n".join(all_contents)

    def _check_history(self, url_or_query):
        header = ''
        for i in range(len(self.history) - 1, -1, -1):  # Start from the last
            if self.history[i][0] == url_or_query:
                header += f"You previously visited this page {round(time.time() - self.history[i][1])} seconds ago.\n"
                return header
        # If it has not appeared before, add it to the history.
        self.history.append((url_or_query, time.time()))
        return header

    def _pre_visit(self, url):
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i][0] == url:
                return f"You previously visited this page {round(time.time() - self.history[i][1])} seconds ago.\n"
        return ""
    
    def _select_api_key_random(self):
        self._last_api_key_index = random.randint(0, len(self.api_key_list) - 1)
        return self.api_key_list[self._last_api_key_index]

    def _select_api_key_with_round_robin(self):
        if not hasattr(self, '_last_api_key_index'):
            self._last_api_key_index = -1
        
        self._last_api_key_index = (self._last_api_key_index + 1) % len(self.api_key_list)
        return self.api_key_list[self._last_api_key_index]
    
    async def _try_api_request(self, api_request_data, api_key, request_data):
        api_headers = {"Content-Type": "application/json", 'X-API-KEY': api_key}
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.serpapi_base_url,
                json=api_request_data,
                headers=api_headers,
                timeout=request_data.get("timeout", SERPER_TIMEOUT),
            )
            response.raise_for_status()
            results = response.json()
            
            # Check if the result is valid
            query = api_request_data.get('q', '')
            filter_year = api_request_data.get('filter_year')
            
            if "organic" not in results:
                return f"No results found for query: '{query}'. Use a less specific query."
            if len(results["organic"]) == 0:
                year_filter_message = f" with filter year={filter_year}" if filter_year is not None else ""
                return f"No results found for '{query}'{year_filter_message}. Try with a more general query, or remove the year filter."
                
            return results

    async def process_request(self, request_data: dict, headers: dict) -> dict:
        api_request_data = request_data.copy()
        querylist = [query.strip() for query in request_data['q'].split('|') if query.strip()]
        if len(querylist) == 0:
            error_messages = f"Query {queries} splitting failed! Please strictly follow the requirements: each query should be separated by '|'!"
            return error_messages

        search_results = {}
        total_results = 0
        remaining_queries = len(querylist)
        serp_num=request_data.get('num')
        remaining_serp = serp_num

        seen_set = set()
        unique_results = []
        for q in querylist:
            try:
                current_serp = max(1, remaining_serp // remaining_queries)
                api_request_data['q'] = q
                api_request_data['num'] = current_serp
                logger.info(f"api_request_data: {api_request_data}")
                snippets = await self.process_request_single(api_request_data, headers)

                
                # Create independent deduplicated results for each query
                unique_results_for_query = []
                if isinstance(snippets, str):
                    unique_results_for_query = [snippets]
                else:
                    # Deduplication processing
                    for result in snippets:
                        if isinstance(result, dict) and 'link' in result:
                            if result['link'] not in seen_set:  # The global seen_set is still used to prevent cross-query duplicates.
                                seen_set.add(result['link'])
                                unique_results_for_query.append(result)
                    
                # Use the independent results of the current query
                search_results[q] = unique_results_for_query[:current_serp]
                total_results += len(search_results[q])
                remaining_serp -= len(search_results[q])
                remaining_queries -= 1
                
            except Exception as e:
                print(f"Error searching for query '{q}': {str(e)}")
                remaining_queries -= 1  # Even if it fails, the remaining query count should be reduced.
        if total_results < serp_num and len(querylist) > 1:
            additional_needed = serp_num - total_results
            # Sort in ascending order by the number of query results, and prioritize supplementing from queries with fewer results.
            sorted_queries = sorted(
                [q for q in querylist if q in search_results and isinstance(search_results[q], list)],
                key=lambda x: len(search_results[x])
            )

            for q in sorted_queries:
                if additional_needed <= 0:
                    break
                current_count = len(search_results[q])
                additional = min(additional_needed, 3)
                try:
                    api_request_data['q'] = q
                    api_request_data['num'] = additional
                    extra_snippets = await self.process_request_single(api_request_data, headers)


                    if isinstance(extra_snippets, str):
                        print(f"Warning: The supplementary results for the query '{q}' returned an error: {extra_snippets}, which has been skipped.")
                        continue

                    if not isinstance(extra_snippets, list):
                        print(f"Warning: The supplementary result format for the query '{q}' is incorrect and has been skipped.")
                        continue

                    if extra_snippets:
                        existing_links = {res['link'] for res in search_results[q] if
                                          isinstance(res, dict) and 'link' in res}

                        new_results = []
                        for res in extra_snippets:
                        
                            if isinstance(res, dict) and 'link' in res:
                                link = res['link']
                                if link not in existing_links:
                                    new_results.append(res)
                            else:
                                print(f"Warning: There are entries with invalid formats in the supplementary results for the query '{q}', which have been skipped.")

                        new_results = new_results[:additional_needed]

                        if new_results:
                            search_results[q].extend(new_results)
                            total_results += len(new_results)
                            additional_needed -= len(new_results)
                except Exception as e:
                    print(f"Error supplementing query '{q}': {str(e)}")

        content = self._to_contents_multiqueries(search_results)

        return content
    async def process_request_single(self, request_data: dict, headers: dict) -> dict:
        """Process Serper API requests, prioritizing retrieval from the cache; if the cache misses, forward the request."""
        start_time = time.time()

        # P0: num logic from README
        original_num = request_data.get("num", 10)
        api_request_data = request_data.copy()

        try:
            # Generate cache key based on the API request (10 or 100 results)
            cache_key = self._generate_cache_key(api_request_data)

            # Check cache
            cached_result_str = self.cache.get(cache_key)
            if cached_result_str:
                self._log_hit_rate(hit=True)
                cached_result = json.loads(cached_result_str)
                # Trim the results to the originally requested number
                if "organic" in cached_result:
                    cached_result["organic"] = cached_result["organic"][:original_num]
                logger.info(
                    f"Request processed from cache in {time.time() - start_time:.2f} seconds."
                )
                results =  cached_result
            else:
                # Cache miss, forward request to Serper API
                self._log_hit_rate(hit=False)
                logger.info(f"Forwarding to Serper API for key: {cache_key}")

                if not self.api_key_list:
                    raise HTTPException(
                        status_code=401, detail="Serper API key not configured"
                    )
                
                api_key = self._select_api_key_random()
                logger.info(f"Choosing api_key: {api_key}")
                try_times = 0

                try:
                    try_times += 1
                    results = await self._try_api_request(api_request_data, api_key, request_data)
                    self.cache.set(cache_key, json.dumps(results))
                    logger.info(f"Successfully cached {results['searchParameters']}")
                except Exception as first_error:
                    logger.warning(f"API Key with ending {api_key[-5:]} failed, trying alternatives")

                    if try_times == len(self.api_key_list):
                        # There are no other available API Keys.
                        logger.warning(f"API Key with ending {api_key[-5:]} failed, no alternatives left")
                        raise first_error
                    
                    # Try the remaining API Keys
                    last_error = first_error
                    while try_times < len(self.api_key_list):
                        try:
                            try_times += 1
                            api_key = self._select_api_key_with_round_robin()
                            logger.info(f"Choosing api_key: {api_key}")
                            results = await self._try_api_request(api_request_data, api_key, request_data)
                            self.cache.set(cache_key, json.dumps(results))
                            logger.info(f"Successfully cached {results['searchParameters']}")
    
                            logger.info(f"Successfully used alternative API Key with ending {api_key[-5:]}")
                            break
                        except Exception as e:
                            logger.warning(f"API Key with ending {api_key[-5:]} failed, trying alternatives")
                            last_error = e
                    else:
                        # All API Keys have failed
                        raise last_error

            web_snippets: List[str] = list()
            idx = 0
            if "organic" in results:
                for page in results["organic"]:
                    idx += 1
                    date_published = ""
                    if "date" in page:
                        date_published = "\nDate published: " + page["date"]

                    source = ""
                    if "source" in page:
                        source = "\nSource: " + page["source"]

                    snippet = ""
                    if "snippet" in page:
                        snippet = "\n" + page["snippet"]

                    _search_result = {
                        "idx": idx,
                        "title": page["title"],
                        "date": date_published,
                        "snippet": snippet,
                        "source": source,
                        "link": page['link']
                    }

                    web_snippets.append(_search_result)
            return web_snippets

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error calling Serper API: {e.response.status_code} {e.response.text}"
            )

            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except Exception as e:
            logger.error(f"Unexpected error during Serper API request: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"An unexpected error occurred: {str(e)}"
            )


# --- FastAPI setting --- #

app = FastAPI(title="Serper API Proxy with Cache")

# --- Application Initialization and Shutdown ---
CACHE_FILE = "server/cache/serper_api_cache_v2.jsonl"  # Shared cache with v2
cache_backend = InMemoryCache(file_path=CACHE_FILE)
proxy_server = SerperProxyServer(cache_backend=cache_backend)

# The atexit registration is now handled within the InMemoryCache class


@app.post("/search")
async def serper_proxy_endpoint(request: Request):
    try:
        # logger.info("TEST---------------")
        logger.info(request)
        request_data = await request.json()
        headers = dict(request.headers)

        return await proxy_server.process_request(request_data, headers)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body.")
    except Exception as e:
        # Unhandled exceptions are caught and logged by the proxy_server now,
        # but we keep a general handler here as a fallback.
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Unhandled exception in endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": time.time(),
        "cache_stats": {
            "total_requests": proxy_server.total_requests,
            "cache_hits": proxy_server.cache_hits,
            "hit_rate": (proxy_server.cache_hits / proxy_server.total_requests * 100) if proxy_server.total_requests > 0 else 0
        }
    }


if __name__ == "__main__":
    # Obtain the host and port from environment variables or default values
    # Read from environment variables first; if they do not exist, use the default values.
    host = os.getenv("SERVER_HOST", None)
    port = os.getenv("WEBSEARCH_PORT", None)
    if port == None:
        raise NotImplementedError("[ERROR] WEBSEARCH_PORT NOT SET")
    port = int(port)

    # Start the server
    logger.info(f"Start the Serper cache server... http://{host}:{port}")
    logger.info(f"Ready to search")
    uvicorn.run(app, host=host, port=port)
