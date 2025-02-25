FROM python:3.12-slim-bookworm
LABEL authors="apduncan"
LABEL org.opencontainers.image.source=https://github.com/apduncan/cvanmf

# Need some build tools for skbio
# Currently we install build-essential which may be overkill
# and could be trimmed down to reduce size of image
RUN apt update && apt -y install build-essential
RUN apt clean
RUN pip install cvanmf --verbose