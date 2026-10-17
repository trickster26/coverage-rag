FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY coverage_rag ./coverage_rag
RUN pip install --no-cache-dir .
COPY data ./data
EXPOSE 8000
CMD ["uvicorn", "coverage_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
