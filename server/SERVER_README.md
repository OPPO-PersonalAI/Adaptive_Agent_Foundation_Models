# Quick Start Guide for Tool Servers

## 📋 Quick Start
Currently, the following three types of tool servers are supported, with key information provided in the table below:

| Server Name      | Core Function Description                                                                 | Main Program File          | Test File                  |
| ---------------- | ---------------------------------------------------------------------------------------- | -------------------------- | -------------------------- |
| serper_cache     | Google Search (with cache) + Multi-API Polling + Result Summarization                     | cache_serper_server.py     | test_cache_serper_server.py|
| crawl_page       | JINA Reader Web Crawling + Multi-API Polling + Content Summarization                       | crawl_page_server.py       | crawl_page_server_test.py  |
| code_exec        | Secure execution of Python code in the nsjail sandbox environment                          | code_execute_server.py     | test_code_execute_server.py|


### ⚠️ Notes
1. Install Dependencies: Run the command below to install the dependency packages listed in `server/requirements.txt`
    ```bash
    cd ./server
    pip install -r requirements.txt
    ```
2. Code Server-related Configuration (required when using the code_exec server or when this function is needed during training):
   - Clone and compile nsjail:
     ```bash
     git clone https://github.com/google/nsjail.git
     cd nsjail
     make
     ```
   - Save absolute path of nsjail:
     ```
     NSJAILPATH="/abs_path/to/your/nsjail/nsjail"
     ```
3. Environment Variable Configuration: Environment variables must be configured in the `servers/.env` file. For a configuration template, refer to the `servers/env_template` file.


### 🚀 Shell Script Operations
The following scripts allow for quick management of starting, stopping, and status checking for all servers. Before operation, you must first navigate to the `server/` directory:

```bash
# 1. Navigate to the server directory (mandatory step)
cd server/

# 2. Stop all running servers
./start_servers.sh stop

# 3. Check the running status of all servers
./start_servers.sh status

# 4. Start all servers
./start_servers.sh start
```


## 🧪 Test Servers
1. Configure environment variables in the `servers/.env` file and execute the `start_servers.sh start` startup script.
2. Run `start_servers.sh test` to test all services.
3. To test a specific server, configure environment variables in `server/version/TestEnvironment.sh` and directly execute the corresponding test script in the `server/server_tests/` directory. Each test script contains specific execution command examples.


## 🔌 Tool Server Interface Details
Below are the interface specifications, request/response formats, and function descriptions for each server, which can be integrated as needed.

### 1. Serper Cache Server (Multi-API Search + Optional JINA Reader Crawling + Summarization)
- **Main Program File**: `cache_serper_server.py`
- **Default Port**: 9002
- **Core Interface**: `POST /search`
- **Function Positioning**: Google Search based on multiple APIs (supports result caching) and automatic AI summary generation
- **Environment Variable Requirement**: The `WEB_SEARCH_SERPER_API_KEY` must be configured in the environment variables. For multiple API keys, separate them with `|` (example: `KEY1|KEY2`)

#### Request Format (JSON)
```json
{
  "q": "Search query content",  // Mandatory field: enter the keywords/question to be searched
  "num": 10                     // Mandatory field: specify the number of search results to return
}
```

#### Response Format (JSON)
```json
{
  "organic": [         // List of search results
    {
      "title": "Page Title",    // Title of the webpage corresponding to the search result
      "link": "https://example.com",  // Webpage link
      "snippet": "Page Summary" // AI-generated summary of the webpage content
    }
  ],
  "searchParameters": {}  // Details of the search request parameters (e.g., query term, number of results)
}
```


### 2. Crawl Page Server (Multi-API Concurrent Webpage Crawling + Summarization)
- **Main Program File**: `crawl_page_server.py`
- **Default Port**: 9000
- **Core Interface**: `POST /crawl_page`
- **Function Positioning**: Concurrent calls to JINA Reader via multiple APIs to crawl specified webpage content, and generate AI summaries based on configurations

#### Request Format (JSON)
```json
{
  "urls": ["https://example.com", "https://example2.com"],  // Mandatory field: list of webpage URLs to crawl
  "web_search_query": "Search query content",              // Mandatory field: associated search keywords (used to optimize summaries)
  "think_content": "Thinking content",                     // Optional field: supplementary contextual description
  "summary_prompt_type": "webthinker_with_goal",           // Optional field: summary prompt type (default value shown below)
  "summary_type": "once",                                  // Optional field: summary method (default value shown below)
  "do_last_summary": false,                                // Optional field: whether to perform a secondary summary on the final result (default: false)
  "chunk_size": 8192,                                      // Optional field: content chunk size for chunk-based summarization (default: 8192 characters)
  "api_url": "API address of the summary model",           // Mandatory field: API endpoint of the summary model
  "api_key": "API key",                                    // Mandatory field: API key for calling the summary model
  "model": "Name of the summary model",                    // Mandatory field: name of the summary model used
  "messages": "Message history"                            // Optional field: conversation history context (for multi-turn summarization)
}
```

#### Key Parameter Description
- **summary_prompt_type (Summary Prompt Type)**:
  - `webthinker_with_goal` (default): Goal-oriented thinking-style prompt
  - `webdancer_with_goal`: Goal-oriented process-style prompt
  - `webthinker`: Basic thinking-style prompt
  - `webdancer`: Basic process-style prompt

- **summary_type (Summary Method)**:
  - `"none"`: Do not generate a summary; return the crawled original content directly (concatenated only)
  - `"once"`: Perform a **one-time aggregated summary** of all crawled content (default)
  - `"chunk"`: Split content according to the configured `chunk_size` and summarize each chunk individually
  - `"page"`: Summarize by URL (one independent summary per webpage)

#### Response Format (JSON)
```json
{
  "success": true,                // Server execution status (true = success, false = failure)
  "obs": "Content after crawling and summarization",  // Core result: text after processing
  "error_message": null,          // Error message (null if successful; specific reason shown if failed)
  "processing_time": 12.34        // Processing time (unit: seconds)
}
```


### 3. Code Exec Server (Python Code Execution)
- **Main Program File**: `code_execute_server.py`
- **Default Port**: 9003
- **Core Interface**: `POST /code_exec`
- **Function Positioning**: Securely execute Python code in the nsjail sandbox environment to avoid local environment contamination

#### Request Format (JSON)
```json
{
  "code_str_list": ["print(1+4)", "print(2-13)"],  // Mandatory field: list of Python code snippets to execute (each element is a segment of code)
  "parameter_list": [],           // Optional field: list of parameters required for code execution (leave empty if no parameters are needed)
}
```

#### Response Format (JSON)
```json
{
  "success": true,                // Overall server execution status (true = server normal, false = server abnormal)
  "obs": [                        // List of code execution details (corresponds to the order of code_str_list)
    [
      true,                       // Execution result of a single code segment (true = success, false = failure)
      "5",              // Code execution output (e.g., results, printed content)
      "[EXECUTED] Code exited with status 0."  // Execution status description (success/error information)
    ],
    [
      true,
      "-11",
      "[EXECUTED] Code exited with status 0."
    ]
  ],
  "error_message": null,          // Server-level error message (null if no error occurs)
  "processing_time": 0.07729768753051758  // Overall processing time (unit: seconds)
}
```