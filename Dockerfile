
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

## Add System Dependencies
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        build-essential \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

## Change working directory
RUN mkdir -p /app/alphafold
WORKDIR /app/alphafold

## Install the AlphaFold3 package + requirements
COPY . .
RUN python -m pip install -r requirements.txt \
    && python -m pip install git+https://github.com/aqlaboratory/openfold \
    && python -m pip install .

