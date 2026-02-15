# Start from the NVIDIA official image (ubuntu-24.04 + cuda-12.9 + python-3.12)
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-06.html
FROM nvcr.io/nvidia/pytorch:25.06-py3

# Define environments
ENV MAX_JOBS=32
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_OPTIONS=""
ENV PIP_ROOT_USER_ACTION=ignore
ENV HF_HUB_ENABLE_HF_TRANSFER="1"
ENV PIP_CONSTRAINT=""

ARG PIP_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Change pip source
RUN pip config set global.index-url "${PIP_INDEX}" && \
    pip config set global.extra-index-url "${PIP_INDEX}" && \
    pip config set global.no-cache-dir "true" && \
    python -m pip install --upgrade pip

# Install systemctl
RUN apt-get update && \
    apt-get install -y -o Dpkg::Options::="--force-confdef" systemd && \
    apt-get clean

# Install libxml2
RUN apt-get update && \
    apt-get install -y libxml2 aria2 git-lfs && \
    apt-get clean

# Uninstall nv-pytorch fork
RUN pip uninstall -y torch torchvision torchaudio \
    pytorch-quantization pytorch-triton torch-tensorrt \
    transformer_engine flash_attn apex \
    xgboost opencv grpcio

# Fix cv2
RUN rm -rf /usr/local/lib/python3.11/dist-packages/cv2

# Install torch (2.9.1 cuda 12.9)
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cu129

# Install vllm
RUN pip install --no-cache-dir "vllm==0.15.1" && pip install torch-memory-saver --no-cache-dir

# Fix packages
RUN pip install --no-cache-dir tensordict torchdata "transformers[hf_xet]==4.57.6" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=19.0.1" pandas \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel==0.6.4 mathruler blobfile xgrammar \
    pytest py-spy pre-commit ruff more_itertools tensorboard gcsfs math-verify langdetect

# Install flash-attn-2.8.3
RUN ABI_FLAG=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')") && \
    URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abi${ABI_FLAG}-cp312-cp312-linux_x86_64.whl" && \
    wget -nv -P /opt/tiger "${URL}" && \
    pip install --no-cache-dir "/opt/tiger/$(basename ${URL})"

# Install AetherEval (always latest main)
RUN git clone --branch main --single-branch https://github.com/SeanLeng1/AetherEval.git /opt/AetherEval && \
    cd /opt/AetherEval && \
    git lfs install && git lfs pull && \
    pip install --no-cache-dir -e /opt/AetherEval

# Install DeepEP
# the dependency of IBGDA
RUN ln -s /usr/lib/x86_64-linux-gnu/libmlx5.so.1 /usr/lib/x86_64-linux-gnu/libmlx5.so

# Clone and build deepep and deepep-nvshmem
RUN git clone -b v2.3.1 https://github.com/NVIDIA/gdrcopy.git && \
    git clone https://github.com/deepseek-ai/DeepEP.git  && \
    cd DeepEP && git checkout a84a248

# Prepare nvshmem
RUN wget https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.2.5-1.txz && \
    tar -xvf nvshmem_src_3.2.5-1.txz && mv nvshmem_src deepep-nvshmem && \
    cd deepep-nvshmem && git apply ../DeepEP/third-party/nvshmem.patch

## Build deepep-nvshmem
RUN apt-get install -y ninja-build cmake

ENV CUDA_HOME=/usr/local/cuda
### Set MPI environment variables. Having errors when not set.
ENV CPATH=/usr/local/mpi/include:$CPATH
ENV LD_LIBRARY_PATH=/usr/local/mpi/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV GDRCOPY_HOME=/workspace/gdrcopy
ENV GDRCOPY_INCLUDE=/workspace/gdrcopy/include


# Reset pip config
RUN pip config unset global.index-url && \
    pip config unset global.extra-index-url
