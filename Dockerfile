# Use an official Python image
FROM python:3.10-slim-bullseye

WORKDIR /application

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy the rest of your code
COPY . .

EXPOSE 5000

CMD ["python", "application.py"]