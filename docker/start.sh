#!/bin/bash
set -e

# RunPod injects PUBLIC_KEY env var
if [ -n "$PUBLIC_KEY" ]; then
    mkdir -p /root/.ssh
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys
fi

# HF login at runtime, not build time
if [ -n "$HF_TOKEN" ]; then
    /root/miniconda3/envs/dualvla/bin/huggingface-cli login --token "$HF_TOKEN"
fi

# Start SSH
service ssh start

# Keep container alive
sleep infinity
