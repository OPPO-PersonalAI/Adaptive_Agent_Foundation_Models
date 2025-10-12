#!/bin/bash

# A²FM Inference Example Script
# This script demonstrates how to set up environment variables and run A²FM inference

echo "A²FM Inference Example"
echo "======================"

# Set required environment variables
# Please replace these with your actual values

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

# JINA API Key (for web search and page crawling)
export JINA_API_KEY="your-jina-api-key-here"

echo "Environment variables set:"
echo "- MODEL_NAME: $MODEL_NAME"
echo "- MODEL_URL: $MODEL_URL"
echo "- OPENAI_API_URL: $OPENAI_API_URL"
echo "- WEBSEARCH_URL: $WEBSEARCH_URL"
echo "- CRAWL_PAGE_URL: $CRAWL_PAGE_URL"
echo "- CODE_EXEC_URL: $CODE_EXEC_URL"
echo ""

# Example 1: Auto mode with default parameters
echo "Example 1: Auto mode with default parameters"
python infer_main.py \
    --input_file ./data/example.json \
    --output_file ./results/auto_output.jsonl \
    --adaptive auto \
    --temperature 1.0 \
    --max_steps_agent 60 \
    --max_steps_reasoning 6 \
    --max_steps_instant 6

echo ""

# Example 2: Force agentic mode for complex tasks
echo "Example 2: Force agentic mode for complex tasks"
python infer_main.py \
    --input_file ./data/complex_tasks.json \
    --output_file ./results/agentic_output.jsonl \
    --adaptive toolcalling_agent \
    --max_steps_agent 100 \
    --temperature 0.8 \
    --web_topk 15

echo ""

# Example 3: Force reasoning mode for analytical tasks
echo "Example 3: Force reasoning mode for analytical tasks"
python infer_main.py \
    --input_file ./data/analytical_tasks.json \
    --output_file ./results/reasoning_output.jsonl \
    --adaptive reasoning_agent \
    --max_steps_reasoning 10 \
    --temperature 0.5

echo ""

# Example 4: Force instant mode for simple tasks
echo "Example 4: Force instant mode for simple tasks"
python infer_main.py \
    --input_file ./data/simple_tasks.json \
    --output_file ./results/instant_output.jsonl \
    --adaptive instant \
    --max_steps_instant 3 \
    --temperature 0.3

echo ""

# Example 5: High-performance parallel processing
echo "Example 5: High-performance parallel processing"
python infer_main.py \
    --input_file ./data/large_dataset.json \
    --output_file ./results/parallel_output.jsonl \
    --adaptive auto \
    --parallel_per_dataset 10 \
    --parallel_on_dataset 8 \
    --max_steps_agent 80

echo ""
echo "All examples completed!"
echo "Check the results directory for output files."
echo ""
echo "Note: Make sure to:"
echo "1. Replace all placeholder values with your actual API keys and URLs"
echo "2. Ensure all tool servers are running before executing inference"
echo "3. Create the necessary input data files in the ./data/ directory"