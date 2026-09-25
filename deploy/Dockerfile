FROM python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MALLOC_ARENA_MAX=2 PORT=8080
WORKDIR /app
COPY deploy/constraints-cpu.txt /app/deploy/constraints-cpu.txt
# The separate CPU index prevents CUDA libraries entering the Linux image.
RUN pip install --no-cache-dir torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -c deploy/constraints-cpu.txt hatchling setuptools \
    && useradd --uid 10001 --create-home evaluator
COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src
RUN pip install --no-cache-dir --no-build-isolation -c deploy/constraints-cpu.txt '.[model]' \
    && python -c 'import torch; assert torch.version.cuda is None'
USER evaluator
EXPOSE 8765
ENTRYPOINT ["rateloop-evaluator"]

FROM runtime AS hosted
COPY deploy/start-hosted.py /app/deploy/start-hosted.py
# Railway volumes initially belong to root. Entrypoint changes only the mount
# directory owner, drops all root groups/UID, then opens private configuration.
USER root
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c 'import os,urllib.request; urllib.request.urlopen("http://127.0.0.1:"+os.environ["PORT"]+"/healthz",timeout=3)'
ENTRYPOINT ["python","deploy/start-hosted.py"]
