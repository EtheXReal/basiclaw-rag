FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV VECTOR_DATA_DIR=/app/data

EXPOSE 7860

CMD ["python", "app_gradio.py"]
