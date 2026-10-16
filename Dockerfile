FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX"
ENV PYTHONUNBUFFERED=1 UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:/root/.local/bin:${PATH}" \
    FORCE_CUDA=1 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential pkg-config openssl iproute2 \
    libgl1 libglib2.0-0 libgomp1 libegl1 libopengl0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /usr/local/bin/uv
RUN uv python install 3.11 && uv venv --python 3.11 /opt/venv
RUN uv pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128 \
    && uv pip install setuptools==70.0.0 wheel ninja numpy==1.26.4 "Cython>=0.29.36,<0.30"
WORKDIR /app
# Keep native dependencies cached when only the panel or documentation changes.
COPY pyproject.toml setup.py /app/
COPY desktop/prepare_build_environment.py /app/desktop/prepare_build_environment.py
COPY thirdparty /app/thirdparty
COPY protocol /app/protocol
COPY mast3r_slam/__init__.py /app/mast3r_slam/__init__.py
COPY mast3r_slam/backend /app/mast3r_slam/backend
RUN python desktop/prepare_build_environment.py /app \
    && uv pip install --no-build-isolation -e /app/thirdparty/mast3r /app/thirdparty/in3d -e /app
COPY mast3r_slam /app/mast3r_slam
COPY config /app/config
COPY resources /app/resources
COPY main.py /app/main.py
COPY LICENSE.md /app/LICENSE.md
COPY scripts/smoke_web_panel.py /app/scripts/smoke_web_panel.py
RUN python -c "import torch, mast3r_slam_backends; from mast3r_slam.web.app import create_app"
# uv's managed interpreter is read-only at runtime; allow the unprivileged
# application user to traverse its installation path.
RUN chmod a+rx /root /root/.local /root/.local/share /root/.local/share/uv \
    && chmod -R a+rX /root/.local/share/uv/python
ENV HOME=/tmp
USER 1000:1000
EXPOSE 8080 8765
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"
CMD ["python", "-m", "mast3r_slam.cli", "web", "--port", "8080"]
