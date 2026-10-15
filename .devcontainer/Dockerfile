# Development container for cvanmf. Built locally by VS Code Dev Containers;
# this image is not published to any registry.
FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

# Build tools needed to compile scikit-bio
RUN apt-get update \
    && apt-get -y install --no-install-recommends build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER vscode

# Poetry for dependency management, Claude Code via the native installer
RUN pipx install poetry \
    && curl -fsSL https://claude.ai/install.sh | bash

ENV PATH="/home/vscode/.local/bin:${PATH}"
