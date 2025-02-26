# Base Image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy local dependencies for offline installation
COPY wheelhouse /wheelhouse
COPY requirements.txt /app/

# Install dependencies from the local wheelhouse
RUN pip install --no-index --find-links=/wheelhouse -r requirements.txt

# Copy the application code
COPY src /app/src

# Set up an entry point for training
ENTRYPOINT ["python", "/app/src/train.py"]