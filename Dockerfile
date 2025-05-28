# Use an official Python image
FROM python:3.13-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY . /app
RUN apt updated -y && apt install awscli

RUN pip install -r requirements.txt

CMD ["python", "app.py"]