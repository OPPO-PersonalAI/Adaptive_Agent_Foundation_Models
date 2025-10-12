# 工具服务器快速启动指南
## 📋 快速启动
支持的工具服务器有：
| 服务器名称 | 描述 | 文件 | 测试 |
| --- | --- | --- | --- |
| serper_cache_v4 | Google Search (with cache) + Multiple API polling + JINA Reader (default no crawl during search) + Summary | cache_serper_server_v4.py | test_cache_serper_server_v4.py |
| crawl_page_v4 | JINA Reader + Multiple API polling + Summary | crawl_page_server_v4.py | crawl_page_server_test_v4.py |
| code_exec | Execute code in nsjail sandbox environment | code_execute_server.py | test_code_execute_server_v4.py |


**注意事项**: 
  1. 请在servers/.env下配置环境变量,参考servers/.env_template


### Shell脚本

```bash
# cd 到server/
cd server/

# help
./start_servers.sh --help

# 启动推理时的默认工具
./start_servers.sh --infer

# 启动训练时的默认工具
./start_servers.sh --train

# 停止所有服务器
./start_servers.sh stop

# 检查服务器状态
./start_servers.sh status

# 启动所有服务器
./start_servers.sh start
```

## 🧪 验证服务器
执行 `start_servers.sh` 脚本后，会自动执行功能测试脚本。如果需要单独测试某个服务器，可以单独执行 `server/server_tests/` 目录下的测试脚本。这些测试脚本内有给出具体的命令示例。

## 工具服务器接口

### 1. Serper Cache Server v4 (多API搜索 + 可选JINA Reader爬取 + 摘要)
- **文件**: `cache_serper_server_v4.py`
- **端口**: 默认 9002
- **接口**: `POST /search`
- **功能**: 多API Google搜索（带缓存）+ 可选JINA Reader爬取 + AI摘要
- **认证**: 请求头不再需要 `X-API-KEY`，请设置环境变量 `WEB_SEARCH_SERPER_API_KEY`，API用'|'分割。Example：`KEY1|KEY2`
- **请求格式**:
  ```json
  {
    "q": "搜索查询"(必须),
    "num": 10(必须)
  }
  ```
- **响应格式**:
  - **不爬取时** (`use_crawl=false`):
    ```json
    {
      "organic": [
        {
          "title": "页面标题",
          "link": "https://example.com",
          "snippet": "页面摘要"
        }
      ],
      "searchParameters": {...}
    }
    ```
  - **爬取时** (`use_crawl=true`): 返回爬取并摘要后的文本内容

### 2. Crawl Page Server v4 (多API并发页面爬取 + 摘要)
- **文件**: `crawl_page_server_v4.py`
- **端口**: 默认 9000
- **接口**: `POST /crawl_page`
- **功能**: 多API并发使用JINA Reader爬取页面并生成AI摘要
- **请求格式**:
  ```json
  {
    "urls": ["https://example.com", "https://example2.com"] (必须),
    "task": "任务描述"(必须),
    "web_search_query": "搜索查询"(必须),
    "think_content": "思考内容"(必须),
    "summary_type": "once"(可选),
    "do_last_summary": false(可选),
    "chunk_size": 8192(可选),
    "api_url": "API地址"(必须),
    "api_key": "API密钥"(必须),
    "model": "模型名称"(必须),
    "messages": "消息历史"(可选)
  }
  ```
- **摘要类型说明**:
  - `"none"`: 不进行摘要，直接连接内容
  - `"once"`: 对所有内容进行一次性摘要
  - `"chunk"`: 按块进行摘要（可配置chunk_size）
  - `"page"`: 按页面进行摘要
- **响应格式**:
  ```json
  {
    "success": true,
    "obs": "爬取并摘要后的内容",
    "error_message": null,
    "processing_time": 12.34
  }
  ```

从轨迹中获取调用 crawl_page 工具需要传入的参数：
```xml
<think> 更早的 think_content </think>
<web_search> 最近的 web_search_query </web_search>
...
<think> 最近的 think_content </think>
<crawl_page> url_1 | url_2 | ... </crawl_page> <---- 调用 crawl_page 工具
```

调用 crawl_page 工具时需要传入：
1. urls - 要爬取的URL列表 = [url_1, url_2, ...]
2. think_content - 最近的思考内容 = "最近的 think_content"
3. web_search_query - 最近的搜索查询 = "最近的 web_search_query"
4. task - 原始问题（暂时不用，可传入占位符）
5. api_url, api_key, model - summary model 的 配置

### 3. Code Exec Server (Python 代码执行)
- **文件**: `code_execute_server.py`
- **端口**: 默认 9003
- **接口**: `POST /code_exec`
- **功能**: 沙盒环境执行Python代码
- **请求格式**:
  - **Code_str** (需传入markdown格式):
    ```json
    {
      "desc": "简单加法" (可选),
      "code_str_list": ["code1", "code2"](必须),
      "parameter_list": [] (可选),
      "task": "测试加法" (可选),
    }
    ```
- **响应格式**:
  ```json
  {
    "success": True or False (服务执行状态),
    "obs": [[True or False (代码执行状态), (代码实际输出), (代码退出状态或报错信息)] (代码1处理结果), [...] (代码2处理结果)], 
    "error_message": (服务器报错信息), 
    "processing_time": (服务器处理时间)
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

