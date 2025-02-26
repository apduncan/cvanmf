FROM ghcr.io/apduncan/cvanmf:latest
LABEL authors="kam24goz"
LABEL org.opencontainers.image.source=https://github.com/apduncan/cvanmf

# Install some extra requirements for the Nextflow pipeline which
# aren't need for a base installation

RUN pip install jupyter papermill
# ps needed by nextflow to produce runtime metrics etc
RUN apt update && apt install -y procps && apt clean