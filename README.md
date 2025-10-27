<div align="center">

<h2>A<sup>2</sup>FM: An Adaptive Agent Foundation Model for Tool-Aware Hybrid Reasoning</h2>
</div>

<div align="center">
  <a href='https://chain-of-agents-afm.github.io/'><img src='https://img.shields.io/badge/Project-Github-red?logo=GitHub&logoWidth=40'></a>
  <a href='https://arxiv.org/abs/2510.12838'><img src='https://img.shields.io/badge/Paper AFM-arXiv-d63031?logo=arxiv&logoColor=white'></a>
  <a href='https://huggingface.co/PersonalAILab/A2FM-32B-rl'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Models-Huggingface-yellow'></a>
</div>

This is the official repository for our paper "A<sup>2</sup>FM: An Adaptive Agent Foundation Model for Tool-Aware Hybrid Reasoning". This is an adaptive branch of [AFM](https://github.com/OPPO-PersonalAI/Agent_Foundation_Models/). Different problems are solved using different modes (agentic, reasoning, instant) to ensure that every token consumption is put to the best use. Our work provides an approach that leverages the accuracy rate in inter-gro2up rollouts within Reinforcement Learning (RL), using the difficulty level of problems to train the model's adaptive capability.

<div align="center">
  <img src="./assets/adaptive_afm.jpg" width="95%" height="auto" />
</div>

# Overview 🎯

A<sup>2</sup>FM presents a **unified framework** that bridges the gap between reasoning-centric and agentic LLMs through adaptive mode selection, achieving superior performance while dramatically reducing costs.

<div align="center">
  <img src="./assets/first_pic.jpg" width="95%" height="auto" />
</div>

## Key Innovations & Highlights

🧠 **Route-then-Align Principle**: A unified framework that bridges reasoning-centric and agentic LLMs through adaptive mode selection, eliminating the inefficiency gap where both families tend to overthink or over-call tools.

⚡ **Three-Mode Architecture**: 
- **Instant Mode**: Direct reasoning for simple tasks (no tool calls)
- **Agentic Mode**: Tool-augmented reasoning for complex problems  
- **Reasoning Mode**: Deep chain-of-thought for analytical tasks

🎯 **Adaptive Policy Optimization (APO)**: The key to training efficient models - enforces adaptive sampling across modes with cost-regularized rewards.

### Training Efficient Models with APO

<div align="center">
  <img src="./assets/rl_comparison.jpg" width="75%" height="auto" />
</div>

**Method**: APO applies cost-regularized rewards and adaptive sampling to optimize mode selection, ensuring every token consumption delivers maximum value.

<div align="center">
  <img src="./assets/rl.jpg" width="95%" height="auto" />
</div>

**Results**: 
- **New SOTA**: 13.4% on BrowseComp, 70.4% on AIME25, 16.7% on HLE
- **45.2% cost reduction** relative to reasoning models, **33.5%** relative to agentic models
- **$0.00487 per correct answer** - substantially higher cost efficiency while maintaining comparable accuracy

# Table of Contents

- [Overview](#overview-)
  - [Key Innovations & Highlights](#key-innovations--highlights)
  - [Training Efficient Models with APO](#training-efficient-models-with-apo)
- [Quick Start](#quick-start)
  - [1. Tool Server Deployment](#1-tool-server-deployment)
  - [2. Model Download & Inference](#2-model-download--inference)
    - [Install Dependencies](#install-dependencies)
    - [Model Download](#model-download)
    - [Deploy Model Server](#deploy-model-server)
    - [Run Inference](#run-inference)
      - [Key Parameters](#key-parameters)
- [Related Work](#related-work)
- [Acknowledgement](#acknowledgement)
  - [Citation](#citation)
- [Star](#star)

# Quick Start

## 1. Tool Server Deployment

Our tool server infrastructure provides **highly stable and fast** tool execution capabilities, which are crucial for both RL training and inference. The system features:

- **🔄 Cache Hit Functionality**: Intelligent caching mechanism reduces redundant API calls and improves response times
- **🛡️ Error Handling & Retry Mechanisms**: Robust error capture and automatic retry logic ensure reliable tool execution
- **⚡ Asynchronous Acceleration**: Multi-threaded and async processing for concurrent tool operations
- **🔧 Multi-API Support**: Fallback mechanisms across multiple API providers for enhanced reliability

Starting tool servers, refer to `./server/SERVER_README.md`:


**Available Tool Servers:**

- **Web Search Server**: Multi-API Google search with intelligent caching
- **Page Crawler Server**: Concurrent page crawling with AI-powered summarization
- **Code Executor Server**: Secure Python code execution in nsjail sandbox


## 2. Model Download & Inference

### Install Dependencies
First, install the required dependencies by executing the command below to install packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Model Download

You can directly download the model by following the links below.

| Model | Download Links | Model Size | Context Length |
| :-----------------: | :-----------------------------------------: | :----------: | :--------------: |
| A<sup>2</sup>FM-32B-rl | [🤗 HuggingFace](https://huggingface.co/PersonalAILab/A2FM-32B-rl) | 32B | 128K |

**Alternative Download Methods:**

1. **Direct from HuggingFace**: Click the 🤗 HuggingFace link above
2. **Script Download**: 
   ```bash
   cd ./model
   python download.py
   ```

### Deploy Model Server

Deploy A<sup>2</sup>FM using vLLM for high-performance inference:

```bash
cd ./deploy
bash ./deploy.sh
```

### Run Inference

**1. Set Environment Variables**

Before running inference, you must set the following required environment variables:

```bash
# Model Configuration
export MODEL_NAME="A2FM-32B-rl"
export MODEL_URL="http://localhost:8000/v1"

# OpenAI API Configuration (for judge and summary models)
export OPENAI_API_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="your-openai-api-key-here"

# Tool Server URLs
export WEBSEARCH_URL="http://localhost:9002"
export CRAWL_PAGE_URL="http://localhost:9000"
export CODE_EXEC_URL="http://localhost:9003"
```

**2. Run Inference**

Prepare your test dataset (refer to `/data/example.json` format) and run inference. The input is .json/.jsonl file and the output is .jsonl file.

```bash
cd ./infer
python infer_main.py --input_file ../data/example.json --output_file ../results/output.jsonl
```

**Quick Start with Example Script:**

```bash
cd ./infer
# Edit example_infer_main.sh to set your actual API keys and URLs
bash example_infer_main.sh
```

#### Key Parameters

**Adaptive Mode Selection (`--adaptive`)**:
- `auto`: Automatic mode selection based on task complexity (recommended)
- `toolcalling_agent`: Force agentic mode with tool usage for complex tasks
- `reasoning_agent`: Force reasoning mode for analytical tasks
- `instant`: Force instant mode for simple tasks (no tool calls)

**Max Steps Configuration** (renamed from retry_attempts for clarity):
- `--max_steps_agent`: Maximum execution steps for agentic mode (default: 60)

**Example Usage**:

```bash
cd ./infer
# Auto mode with custom parameters
python infer_main.py \
    --input_file ../data/example.json \
    --output_file ../results/output.jsonl \
    --adaptive auto \
    --max_steps_agent 60 \
    --temperature 1.0 \
    --parallel_per_dataset 5

# Force agentic mode
python infer_main.py \
    --input_file ../data/example.json \
    --output_file ../results/agentic_output.jsonl \
    --adaptive toolcalling_agent \
    --max_steps_agent 100

# Force instant mode
python infer_main.py \
    --input_file ../data/example.json \
    --output_file ../results/instant_output.jsonl \
    --adaptive instant 
```

**Help**: Run `python infer_main.py --help` for complete parameter list.

# Related Work
Listed below are friendly links to relevant agents works from OPPO PersonalAI Lab:

- [Flash-Searcher](https://github.com/OPPO-PersonalAI/Flash-Searcher): Fast and Effective Web Agents via DAG-Based Parallel Execution
- [Agent Foundation Models](https://github.com/OPPO-PersonalAI/Agent_Foundation_Models): Chain-of-Agents: End-to-End Agent Foundation Models via Multi-Agent Distillation and Agentic RL
- [TaskCraft](https://github.com/OPPO-PersonalAI/TaskCraft): Automated Generation of Agentic Tasks
- [OAgents](https://github.com/OPPO-PersonalAI/OAgents): An Empirical Study of Building Effective Agents
- [Agent-KB](https://github.com/OPPO-PersonalAI/Agent-KB): Leveraging Cross-Domain Experience for Agentic Problem Solving
- [MiCoTA](https://github.com/OPPO-PersonalAI/MiCoTA): Bridging the Learnability Gap with Intermediate CoT and Teacher Assistants

# Acknowledgement

We would like to express our sincere gratitude to the original authors and contributors of LLaMA-Factory and verl, an excellent open-source project that provided a solid foundation for our work. Our implementation has been adapted from the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [verl](https://github.com/volcengine/verl).

## Citation

If you find A<sup>2</sup>FM useful in your research or applications, we would appreciate it if you could cite our work:

```bibtex
@article{chen2025textsuperscript,
  title={A$\backslash$textsuperscript $\{$2$\}$ FM: An Adaptive Agent Foundation Model for Tool-Aware Hybrid Reasoning},
  author={Chen, Qianben and Cao, Jingyi and Zhang, Jiayu and Qin, Tianrui and Li, Xiaowan and Zhu, King and Shi, Dingfeng and Zhu, He and Liu, Minghao and Liang, Xiaobo and others},
  journal={arXiv preprint arXiv:2510.12838},
  year={2025}
}
```

# Star

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=OPPO-PersonalAI/Adaptive_Agent_Foundation_Models&type=Date)](https://github.com/OPPO-PersonalAI/Adaptive_Agent_Foundation_Models)

</div>
