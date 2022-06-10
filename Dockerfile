FROM continuumio/miniconda3:latest AS build
# ref https://pythonspeed.com/articles/conda-docker-image-size/
# archive https://web.archive.org/web/20220426133333/https://pythonspeed.com/articles/conda-docker-image-size/

# install docker on do machines
# curl -fsSL https://get.docker.com -o get-docker.sh
# sudo sh get-docker.sh

# dockerhub: joennlae/halutmatmul-conda-gpu

SHELL ["/bin/bash", "--login", "-c"]

# Create the environment:
COPY environment_lock.yml .
RUN conda env create --file environment_lock.yml --name halutmatmul_gpu

# Install conda-pack:
RUN conda install -c conda-forge conda-pack

# Use conda-pack to create a standalone enviornment
# in /venv:
RUN conda-pack -n halutmatmul_gpu  -o /tmp/env.tar && \
  mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
  rm /tmp/env.tar

# We've put venv in same path it'll be in final image,
# so now fix up paths:
RUN /venv/bin/conda-unpack


# The runtime-stage image; we can use Debian as the
# base image since the Conda env also includes Python
# for us.
FROM ubuntu:16.04 AS runtime

# Copy /venv from the previous stage:
COPY --from=build /venv /venv

RUN apt-get -y update
RUN apt-get -y install software-properties-common
RUN apt-add-repository -y ppa:git-core/ppa
RUN apt-get -y update
RUN apt-get -y install git

RUN echo "source /venv/bin/activate" >> ~/.bashrc
SHELL ["/bin/bash", "-c"]
