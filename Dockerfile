FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y awscli && apt-get clean

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port
EXPOSE 8080

# Run the application with Gunicorn
CMD ["gunicorn", "--workers=4", "--bind", "0.0.0.0:8080", "app:app"]
