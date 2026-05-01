FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ARG LOCAL_DIR=/opt/ml/code
ARG REQUIREMENTS=requirements.txt

### useful only when using torch.compile ###
### vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv ###
# RUN sed -i 's|http://|https://|g' /etc/apt/sources.list && \
#     apt-get update && \
#     apt-get install -y --no-install-recommends build-essential && \
#     rm -rf /var/lib/apt/lists/*

WORKDIR $LOCAL_DIR

# Install git (needed for pip install from git repositories)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# copy stuff that should not change too often
COPY .project-root .
COPY $REQUIREMENTS .



RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# installs (TODO: use uv?)
RUN uv pip install --system --upgrade pip && \
    uv pip install --system --no-cache-dir -r requirements.txt



COPY pretrained/ pretrained/
COPY configs/ configs/
COPY gdr/ gdr/

# Set HuggingFace token as environment variable from secret
# This allows models to authenticate when loading from gated repos
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# ENTRYPOINT for local debug
ENTRYPOINT ["python", "gdr/train.py"]
