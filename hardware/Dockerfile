FROM condaforge/mambaforge:latest AS build
# ref https://pythonspeed.com/articles/conda-docker-image-size/
# archive https://web.archive.org/web/20220426133333/https://pythonspeed.com/articles/conda-docker-image-size/

# install docker on do machines
# curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh

# dockerhub: joennlae/halutmatmul-conda-hw
# ghcr: ghcr.io/joennlae/halutmatmul-conda-hw
# https://github.com/features/packages

SHELL ["/bin/bash", "--login", "-c"]

# Create the environment:
COPY environment_hw.yml .
RUN mamba env create --file environment_hw.yml --name halutmatmul_hw

# Install conda-pack:
RUN conda install -c conda-forge conda-pack

# Use conda-pack to create a standalone enviornment
# in /venv:
RUN conda-pack -n halutmatmul_hw  -o /tmp/env.tar && \
  mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
  rm /tmp/env.tar

# We've put venv in same path it'll be in final image,
# so now fix up paths:
RUN /venv/bin/conda-unpack


# The runtime-stage image; we can use Debian as the
# base image since the Conda env also includes Python
# for us.
FROM ubuntu:22.04 AS runtime

# Copy /venv from the previous stage:
COPY --from=build /venv /venv

RUN apt-get -y update
RUN apt-get -y install software-properties-common
RUN apt-add-repository -y ppa:git-core/ppa
RUN apt-get -y update
RUN apt-get -y install git make build-essential time libglu1
RUN apt install -y curl python3 cmake libglu1-mesa-dev

# bender
ARG BENDER_VERSION=0.27.1
RUN mkdir -p /tools/bender && chmod 777 /tools/bender && \
    cd /tools/bender && curl --proto '=https' --tlsv1.2 -sSf https://pulp-platform.github.io/bender/init | bash -s -- ${BENDER_VERSION}

ENV PATH="${PATH}:/tools/bender"

# slang
ARG SLANG_VERSION=v4.0
RUN mkdir -p /tools/slang && chmod 777 /tools/slang && \
    cd /tools/slang && git clone https://github.com/MikePopoloski/slang.git && \
    cd slang && git checkout ${SLANG_VERSION} && \
    cmake -B build && cmake --build build -j$(nproc) && \
    cmake --install build --strip

ENV PATH="${PATH}:/tools/slang"

# svase
ARG SVASE_VERSION=master
RUN mkdir -p /tools/svase && chmod 777 /tools/svase && \
    cd /tools/svase && git clone https://github.com/pulp-platform/svase.git && \
    cd svase && git checkout ${SVASE_VERSION} && \
    make all

ENV PATH="${PATH}:/tools/svase/build"

# morty from master via cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y
RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc
ENV PATH="/root/.cargo/bin:${PATH}"
RUN cargo install --git https://github.com/pulp-platform/morty.git


RUN echo "source /venv/bin/activate" >> ~/.bashrc
SHELL ["/bin/bash", "-c"]
