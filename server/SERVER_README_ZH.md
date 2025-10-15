# 工具服务器快速启动指南

## 📋 快速启动
当前支持以下三类工具服务器，核心信息如下表所示：

| 服务器名称       | 核心功能描述                                                                 | 主程序文件                | 测试文件                  |
| ---------------- | ---------------------------------------------------------------------------- | ------------------------- | ------------------------- |
| serper_cache     | Google Search（带缓存） + 多API轮询 + 结果摘要                               | cache_serper_server.py    | test_cache_serper_server.py|
| crawl_page       | JINA Reader网页爬取 + 多API轮询 + 内容摘要                                   | crawl_page_server.py      | crawl_page_server_test.py |
| code_exec        | 在nsjail沙箱环境中安全执行Python代码                                          | code_execute_server.py    | test_code_execute_server.py|


### ⚠️ 注意事项
1. 安装依赖：执行命令安装 `server/requirements.txt` 中的依赖包
    ```bash
    cd ./server
    pip install -r requirements.txt
    ```
2. Code Server 相关配置（使用code_exec服务器或训练过程中需用到此功能时）：
   - 克隆并编译 nsjail：
     ```bash
     git clone https://github.com/google/nsjail.git
     cd nsjail
     make
     ```
   - 环境变量需要添加 nsjail 绝对路径：
     ```
     NSJAILPATH="/abs_path/to/your/nsjail/nsjail"
     ```
3. 环境变量配置：需在 `servers/.env` 文件中配置环境变量，配置模板可参考 `servers/env_template` 文件。


### 🚀 Shell脚本操作
通过以下脚本可快速管理所有服务器的启停与状态查询，操作前需先进入 `server/` 目录：

```bash
# 1. 进入server目录（必做步骤）
cd server/

# 2. 停止所有已启动的服务器
./start_servers.sh stop

# 3. 检查所有服务器的运行状态
./start_servers.sh status

# 4. 启动所有服务器
./start_servers.sh start
```


## 🧪 验证服务器
1. 需在 `servers/.env` 文件中配置环境变量，执行 `start_servers.sh start` 启动脚本。
2. 执行`start_servers.sh test`测试所有服务。
3. 若需单独测试某一个服务器，配置`server/version/TestEnvironment.sh`中的环境变量，可直接执行 `server/server_tests/` 目录下对应的测试脚本，每个测试脚本内均包含具体的执行命令示例。


## 🔌 工具服务器接口详情
以下是各服务器的接口规格、请求/响应格式及功能说明，可根据需求对接使用。

### 1. Serper Cache Server（多API搜索 + 可选JINA Reader爬取 + 摘要）
- **主程序文件**: `cache_serper_server.py`
- **默认端口**: 9002
- **核心接口**: `POST /search`
- **功能定位**: 基于多API的Google搜索（支持结果缓存），并自动生成AI摘要
- **环境变量要求**: 需在环境变量中配置 `WEB_SEARCH_SERPER_API_KEY`，多API密钥用 `|` 分隔（示例：`KEY1|KEY2`）

#### 请求格式（JSON）
```json
{
  "q": "搜索查询内容",  // 必选字段，填写需要搜索的关键词/问题
  "num": 10           // 必选字段，指定返回的搜索结果数量
}
```

#### 响应格式（JSON）
```json
{
  "organic": [         // 搜索结果列表
    {
      "title": "页面标题",    // 搜索结果对应的网页标题
      "link": "https://example.com",  // 网页链接
      "snippet": "页面摘要"   // 网页内容的AI摘要
    }
  ],
  "searchParameters": {}  // 搜索请求的参数详情（如查询词、结果数量等）
}
```


### 2. Crawl Page Server（多API并发页面爬取 + 摘要）
- **主程序文件**: `crawl_page_server.py`
- **默认端口**: 9000
- **核心接口**: `POST /crawl_page`
- **功能定位**: 多API并发调用JINA Reader爬取指定网页内容，并根据配置生成AI摘要

#### 请求格式（JSON）
```json
{
  "urls": ["https://example.com", "https://example2.com"],  // 必选字段，需爬取的网页URL列表
  "web_search_query": "搜索查询内容",                        // 必选字段，关联的搜索关键词（用于优化摘要）
  "think_content": "思考内容",                              // 可选字段，补充上下文说明
  "summary_prompt_type": "webthinker_with_goal",            // 可选字段，摘要提示词类型（默认值如下）
  "summary_type": "once",                                   // 可选字段，摘要方式（默认值如下）
  "do_last_summary": false,                                 // 可选字段，是否对最终结果二次摘要（默认false）
  "chunk_size": 8192,                                       // 可选字段，按块摘要时的内容块大小（默认8192字符）
  "api_url": "summary model 的API地址",                     // 必选字段，摘要模型的API接口地址
  "api_key": "API密钥",                                     // 必选字段，调用摘要模型的API密钥
  "model": "summary模型名称",                               // 必选字段，使用的摘要模型名称
  "messages": "消息历史"                                    // 可选字段，对话历史上下文（用于多轮摘要）
}
```

#### 关键参数说明
- **summary_prompt_type（摘要提示词类型）**:
  - `webthinker_with_goal`（默认）：带目标导向的思考型提示词
  - `webdancer_with_goal`：带目标导向的流程型提示词
  - `webthinker`：基础思考型提示词
  - `webdancer`：基础流程型提示词

- **summary_type（摘要方式）**:
  - `"none"`：不生成摘要，直接返回爬取的原始内容（仅拼接）
  - `"once"`：对所有爬取内容进行**一次性汇总摘要**（默认）
  - `"chunk"`：按配置的 `chunk_size` 分割内容，对每个块单独摘要
  - `"page"`：按URL分页面摘要，每个网页生成一个独立摘要

#### 响应格式（JSON）
```json
{
  "success": true,                // 服务执行状态（true=成功，false=失败）
  "obs": "爬取并摘要后的内容",     // 核心结果，包含处理后的文本
  "error_message": null,          // 错误信息（成功时为null，失败时显示具体原因）
  "processing_time": 12.34        // 处理耗时（单位：秒）
}
```


### 3. Code Exec Server（Python代码执行）
- **主程序文件**: `code_execute_server.py`
- **默认端口**: 9003
- **核心接口**: `POST /code_exec`
- **功能定位**: 在nsjail沙箱环境中安全执行Python代码，避免本地环境污染

#### 请求格式（JSON）
```json
{
  "code_str_list": ["print(1+4)", "print(2-13)"],// 必选字段，需执行的Python代码列表（每个元素为一段代码）
  "parameter_list": [],           // 可选字段，代码执行所需的参数列表（如无参数可留空）
}
```

#### 响应格式（JSON）
```json
{
  "success": true,                // 服务器整体执行状态（true=服务正常，false=服务异常）
  "obs": [                        // 代码执行详情列表（与code_str_list顺序对应）
    [
      true,                       // 单段代码执行结果（true=成功，false=失败）
      "5",                        // 代码执行输出（如结果、打印内容）
      "[EXECUTED] Code exited with status 0."  // 执行状态说明（成功/报错信息）
    ],
    [
      true,
      "-11",
      "[EXECUTED] Code exited with status 0."
    ]
  ],
  "error_message": null,          // 服务器级错误信息（如无错误则为null）
  "processing_time": 0.07729768753051758  // 整体处理耗时（单位：秒）
}
```