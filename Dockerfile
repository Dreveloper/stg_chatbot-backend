# Build stage
FROM python:3.13-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.13-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY . .

RUN mkdir -p /app/data/chroma

EXPOSE 10000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "10000"]