FROM python:3.9

WORKDIR /app

RUN pip install mlflow

CMD ["mlflow", "ui", "--host", "0.0.0.0"]