#!/bin/bash
export TORCHDYNAMO_VERBOSE=1
export OMP_NUM_THREADS=16
export VLLM_USE_V1=1

model_path=Your_model_path
base_modelname=Your_model_name
base_port=1

### customize your own config ###
### This is just a reference (recommendation) ###
# Deployment parameters
INSTANCES=1                # Number of deployment instances (1 deployments)
GPUS_PER_INSTANCE=2        # Number of GPUs per instance (2 cards)
max_model_len=131072       # Maximum model length
LOG_DIR="log_dir"          # Log directory
WAIT_TIMEOUT=300           # Timeout for waiting server startup (seconds)

# Create log directory
mkdir -p "$LOG_DIR"

# Get net0 IP address
net0_ip=$(ifconfig net0 | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n 1)

# If net0 doesn't exist, try to get other main network card IP
if [ -z "$net0_ip" ]; then
    net0_ip=$(hostname -I | awk '{print $1}')
fi

# Process dots in IP
ip_sanitized=$(echo "$net0_ip" | tr '.' '_')

# Build log filename prefix
log_prefix="${base_modelname}_${ip_sanitized}"

# Deploy multiple instances of the specified model
echo "Starting deployment of ${INSTANCES} instances, model: ${base_modelname}"

for ((i=0; i<INSTANCES; i++)); do
    # Calculate GPUs used by current instance
    start_gpu=$((i * GPUS_PER_INSTANCE))
    end_gpu=$((start_gpu + GPUS_PER_INSTANCE - 1))
    gpu_list=$(seq $start_gpu $end_gpu | tr '\n' ',')
    gpu_list=${gpu_list%,}

    # Calculate port for current instance
    port=$((base_port + i))
    
    # Build instance name and log file path
    instance_name="${base_modelname}_inst${i}"
    log_file="${LOG_DIR}/${log_prefix}_inst${i}.log"
    
    # Start service instance
    echo "Starting instance ${instance_name}: port ${port}, using GPU ${gpu_list}"
    
    # --rope-scaling '{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":32768}' \
    nohup bash -c "
        export CUDA_VISIBLE_DEVICES=${gpu_list}
        vllm serve ${model_path} \
            --served-model-name ${base_modelname} \
            --max-model-len ${max_model_len} \
            --max-seq-len ${max_model_len} \
            --rope-scaling '{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":32768}' \
            --tensor-parallel-size ${GPUS_PER_INSTANCE} \
            --gpu-memory-utilization 0.9 \
            --max-num-seqs 32 \
            --enable-prefix-caching \
            --trust-remote-code \
            --uvicorn-log-level debug \
            --host 0.0.0.0 \
            --port ${port}
    " > "$log_file" 2>&1 &
    
    # Wait for server to complete startup
    echo "Waiting for instance ${instance_name} to start..."
    start_time=$(date +%s)
    server_started=0
    
    while [ $(( $(date +%s) - start_time )) -lt $WAIT_TIMEOUT ]; do
        if grep -q "INFO:     Started server process" "$log_file"; then
            server_started=1
            break
        fi
        sleep 5
    done
    
    if [ $server_started -eq 1 ]; then
        echo "Instance ${instance_name} started successfully"
    else
        echo "Warning: Instance ${instance_name} did not start successfully within ${WAIT_TIMEOUT} seconds"
    fi
done

# Output deployment completion information and access addresses
echo -e "\n===== Deployment Complete ====="
echo "Server IP (net0): $net0_ip"
echo "Accessible URL list:"
for ((i=0; i<INSTANCES; i++)); do
    port=$((base_port + i))
    echo "  - http://$net0_ip:$port"
done
echo "===================="
echo "Log file path: $LOG_DIR"
echo "Log file naming format: ${base_modelname}_${ip_sanitized}_inst<number>.log"
