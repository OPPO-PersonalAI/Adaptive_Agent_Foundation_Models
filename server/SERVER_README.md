# Tool Server Quick Start Guide
## 📋 Quick Start
Supported tool servers:
| Server Name | Description | File | Test |
| --- | --- | --- | --- |
| serper_cache_v4 | Google Search (with cache) + Multiple API polling + JINA Reader (default no crawl during search) + Summary | cache_serper_server_v4.py | test_cache_serper_server_v4.py |
| crawl_page_v4 | JINA Reader + Multiple API polling + Summary | crawl_page_server_v4.py | crawl_page_server_test_v4.py |
| code_exec | Execute code in nsjail sandbox environment | code_execute_server.py | test_code_execute_server_v4.py |

**Notes**: 
  1. Please configure environment variables in servers/.env, refer to servers/.env_template


### Shell Scripts

```bash
# cd to server/
cd server/

# help
./start_servers.sh --help

# Start default tools for inference
./start_servers.sh --infer

# Start default tools for training
./start_servers.sh --train

# Stop all servers
./start_servers.sh stop

# Check server status
./start_servers.sh status

# Start all servers
./start_servers.sh start
```

## 🧪 Verify Servers
After executing the `start_servers.sh` script, functional test scripts will run automatically. If you need to test a specific server individually, you can execute the test scripts in the `server/server_tests/` directory. These test scripts provide specific command examples.

## Tool Server Interfaces

### 1. Serper Cache Server v4 (Multiple API Search + Optional JINA Reader Crawling + Summary)
- **File**: `cache_serper_server_v4.py`
- **Port**: Default 9002
- **Interface**: `POST /search`
- **Function**: Multiple API Google Search (with cache) + Optional JINA Reader Crawling + AI Summary
- **Authentication**: Request header no longer needs `X-API-KEY`, please set the environment variable `WEB_SEARCH_SERPER_API_KEY`, APIs separated by '|'. Example: `KEY1|KEY2`
- **Request Format**:
  ```json
  {
    "q": "search query"(required),
    "num": 10(required)
  }
  ```
- **Response Format**:
  - **Without crawling** (`use_crawl=false`):
    ```json
    {
      "organic": [
        {
          "title": "Page title",
          "link": "https://example.com",
          "snippet": "Page summary"
        }
      ],
      "searchParameters": {...}
    }
    ```
  - **With crawling** (`use_crawl=true`): Returns crawled and summarized text content

### 2. Crawl Page Server v4 (Multiple API Concurrent Page Crawling + Summary)
- **File**: `crawl_page_server_v4.py`
- **Port**: Default 9000
- **Interface**: `POST /crawl_page`
- **Function**: Multiple API Concurrent JINA Reader Page Crawling and AI Summary Generation
- **Request Format**:
  ```json
  {
    "urls": ["https://example.com", "https://example2.com"] (required),
    "task": "task description"(required),
    "web_search_query": "search query"(required),
    "think_content": "thinking content"(required),
    "summary_type": "once"(optional),
    "do_last_summary": false(optional),
    "chunk_size": 8192(optional),
    "api_url": "API address"(required),
    "api_key": "API key"(required),
    "model": "model name"(required),
    "messages": "message history"(optional)
  }
  ```
- **Summary Type Description**:
  - `"none"`: No summarization, directly connect content
  - `"once"`: One-time summarization of all content
  - `"chunk"`: Summarize by chunks (configurable chunk_size)
  - `"page"`: Summarize by page
- **Response Format**:
  ```json
  {
    "success": true,
    "obs": "crawled and summarized content",
    "error_message": null,
    "processing_time": 12.34
  }
  ```

Getting parameters for calling the crawl_page tool from the trace:
```xml
Extracting parameters needed for calling the crawl_page tool from the trajectory:
```xml
<think> earlier think_content </think>
<web_search> most recent web_search_query </web_search>
...
<think> most recent think_content </think>
<crawl_page> url_1 | url_2 | ... </crawl_page> <---- calling the crawl_page tool
```

Parameters needed when calling the crawl_page tool:
1. urls - List of URLs to crawl = [url_1, url_2, ...]
2. think_content - Most recent thinking content = "most recent think_content"
3. web_search_query - Most recent search query = "most recent web_search_query"
4. task - Original question (not needed for now, can pass a placeholder)
5. api_url, api_key, model - summary model configuration

### 3. Code Exec Server (Python Code Execution)
- **File**: `code_execute_server.py`
- **Port**: Default 9003
- **Endpoint**: `POST /code_exec`
- **Function**: Execute Python code in a sandbox environment
- **Request Format**:
  - **Code_str** (needs to be in markdown format):
    ```json
    {
      "desc": "Simple addition" (optional),
      "code_str_list": ["code1", "code2"](required),
      "parameter_list": [] (optional),
      "task": "Test addition" (optional),
    }
    ```
- **Response Format**:
  ```json
  {
    "success": True or False (service execution status),
    "obs": [[True or False (code execution status), (actual code output), (code exit status or error message)] (code 1 result), [...] (code 2 result)], 
    "error_message": (server error message), 
    "processing_time": (server processing time)
  }
  ```
- **Example** :
  ```json
  {
    "success": True,
    "obs": [[True, '[OUTPUT]: 5', '[EXECUTED] Code exited with status 0.'], [True, '[OUTPUT]: -11', '[EXECUTED] Code exited with status 0.']], 
    "error_message": None, 
    "processing_time": 0.07729768753051758
  }
  ```