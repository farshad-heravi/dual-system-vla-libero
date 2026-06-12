FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/root
ENV PATH="/root/miniconda3/bin:$PATH"
ENV MUJOCO_GL=egl
ENV MUJOCO_EGL_DEVICE_ID=0
ENV CUDA_VISIBLE_DEVICES=0
ENV PYOPENGL_PLATFORM=egl

# System deps + SSH
RUN apt-get update && apt-get install -y \
    wget curl git \
    cmake build-essential python3-dev pkg-config \
    libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libswscale-dev libswresample-dev libavfilter-dev \
    libgl1-mesa-glx libegl1-mesa libgles2-mesa-dev \
    libegl1-mesa-dev libglew-dev mesa-utils \
    openssh-server \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir /var/run/sshd

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /root/miniconda3 \
    && /root/miniconda3/bin/conda init bash \
    && rm miniconda.sh

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda create -n dualvla python=3.10 -y

RUN conda run -n dualvla pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

RUN conda run -n dualvla pip install "lerobot[libero]"

RUN conda run -n dualvla pip install \
    "git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"

RUN conda run -n dualvla pip install qwen-vl-utils accelerate huggingface_hub

# flash-attn isolated -- fails loudly without breaking other layers
RUN conda run -n dualvla pip install flash-attn --no-build-isolation

# Auto-activate env for interactive SSH sessions
RUN echo "conda activate dualvla" >> /root/.bashrc

# Startup script: injects SSH pubkey from RunPod env var, starts SSH, keeps container alive
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 22

CMD ["/start.sh"]
