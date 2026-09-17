FROM python:3.12-slim
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
WORKDIR /app
COPY . /app
# Build downloads software only. Provision model weights explicitly outside runtime.
RUN pip install --no-cache-dir '.[model]' && useradd --uid 10001 --create-home evaluator
USER evaluator
EXPOSE 8765
ENTRYPOINT ["rateloop-evaluator"]
