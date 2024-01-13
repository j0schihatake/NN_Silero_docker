# Dockerfile to deploy a llama-cpp container with conda-ready environments

# docker pull continuumio/miniconda3:latest

ARG TAG=latest
FROM continuumio/miniconda3:$TAG

RUN apt-get update \
    && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        git \
        uvicorn \
        libportaudio2 \
        locales \
        sudo \
        build-essential \
        dpkg-dev \
        ca-certificates \
        netbase\
        tzdata \
        nano \
        software-properties-common \
        python3-venv \
        python3-tk \
        pip \
        bash \
        ncdu \
        net-tools \
        openssh-server \
        libglib2.0-0 \
        libsm6 \
        libgl1 \
        libxrender1 \
        libxext6 \
        ffmpeg \
        wget \
        curl \
        psmisc \
        rsync \
        vim \
        unzip \
        htop \
        pkg-config \
        libcairo2-dev \
        libgoogle-perftools4 libtcmalloc-minimal4  \
    && rm -rf /var/lib/apt/lists/*

# Setting up locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8

# Create user:
RUN groupadd --gid 1020 silero-group
RUN useradd -rm -d /home/silero-user -s /bin/bash -G users,sudo,silero-group -u 1000 silero-user

RUN python3 -m pip install torch torchvision torchaudio

# Необходимые зависимости:
RUN pip3 install git+https://github.com/Yuego/num2t4ru
RUN pip3 install gradio
RUN pip3 install omegaconf
RUN pip3 install argparse
RUN pip3 install starlette

# Устанавливаем модуль для озвучивания текста:
RUN python3 -m pip install silero

# FastApi
RUN python3 -m pip install pydantic uvicorn[standard] fastapi

RUN python3 -m pip install torch torchvision torchaudio

# Update user password:
RUN echo 'silero-user:admin' | chpasswd

RUN mkdir /home/silero-user/silero

RUN mkdir /home/silero-user/silero/output

RUN mkdir /home/silero-user/silero/src

RUN mkdir /home/silero-user/silero/model

RUN cd /home/silero-user/silero

# Тут переместить app.py в корень (для fastapi, все переезжает в папку до src)
ADD src/fast.py /home/silero-user/silero/src
ADD src/tts.py /home/silero-user/silero/

# Preparing for login
RUN chmod 777 /home/silero-user/silero/output
RUN chmod 777 /home/silero-user/silero
ENV HOME /home/silero-user/silero/
WORKDIR ${HOME}
USER silero-user

CMD uvicorn src.fast:app --host 0.0.0.0 --port 8083 --reload

# Docker:
# docker build -t silero .
# docker run -it -dit --name silero -p 8083:8083 -v D:/Develop/NeuronNetwork/Silero/NN_Silero_docker/temp/:/home/silero-user/silero/output/  --gpus all --restart unless-stopped silero:latest
# docker run -it -dit --name silero -p 8083:8083 -v D:/Develop/NeuronNetwork/Silero/NN_Silero_docker/temp/:/home/silero-user/silero/output/ --restart unless-stopped silero:latest

# Debug:
# docker container attach silero