FROM nvidia/cuda:11.3.1-base-ubuntu20.04

ENV PYTHON_VERSION=3.9

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qq update \
    && apt-get -qq install --no-install-recommends \
    git \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    ffmpeg \
    libopencv-dev \
    python3-opencv \
    #curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -s -f /usr/bin/pip3 /usr/bin/pip

RUN pip install --upgrade pip

RUN pip install onnxruntime==1.14.1 safetensors==0.3.0 transformers==4.27.3 xformers==0.0.16 

RUN pip install omegaconf cutlass triton opencv-python

RUN pip install  torch==2.0.0+cu117  torchvision=0.15.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

RUN git clone https://github.com/huggingface/diffusers.git

WORKDIR /diffusers

RUN pip install -e ".[torch]"

ENV USE_TORCH=1

WORKDIR /home

COPY docker-entrypoint.py /home
COPY v1-inference.yaml /home

#ENTRYPOINT [ "docker-entrypoint.py" ]
