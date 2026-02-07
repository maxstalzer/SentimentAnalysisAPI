FROM python:3.11-slim

WORKDIR /app

# 1. Install 'uv'
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 2. Copy the quantized model folder
COPY tiny_model_onnx /app/tiny_model_onnx

# 3. Install Minimal Dependencies
# We install 'transformers' but WITHOUT torch/tensorflow to save massive space.
# 'optimum[onnxruntime]' pulls in onnxruntime which is small.
RUN uv pip install --system --no-cache \
    fastapi \
    uvicorn \
    numpy \
    "optimum[onnxruntime]" \
    transformers

# 4. Copy Code
COPY src /app/src

EXPOSE 8000

CMD ["uvicorn", "src.sentiment_analysis_api.main:app", "--host", "0.0.0.0", "--port", "8000"]