FROM tiangolo/uvicorn-gunicorn-fastapi:latest

COPY ./app /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
