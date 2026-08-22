FROM python:3.14-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN pip install uv

RUN uv sync --frozen

COPY src/ ./src/
COPY models/ ./models/
COPY start.sh .

RUN chmod +x start.s

CMD ["./start.sh"]