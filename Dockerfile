# Use the official Python 3.11 slim image.
FROM python:3.11-slim

# Prevent Python from writing .pyc files and force stdout/stderr to be unbuffered.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container.
WORKDIR /app

# Install system dependencies including build tools and Python development headers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container.
COPY requirements.txt .

# Upgrade pip and install Python dependencies.
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the project code into the container.
COPY . .

# Expose the port your Flask app will run on.
EXPOSE 8000

# Set the command to run the Flask application.
CMD ["python", "app.py"]