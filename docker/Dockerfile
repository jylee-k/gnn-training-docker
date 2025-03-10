# Base Image
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# Install Python 3.12
RUN apt-get update && apt-get install -y python3.12 python3-pip

# Set Python 3.12 as the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Set working directory
WORKDIR /app

# # Copy local dependencies for offline installation
# COPY wheelhouse /wheelhouse

# # Install dependencies from the local wheelhouse
# RUN pip install --no-index --no-deps --break-system-packages /wheelhouse/* && rm -rf /wheelhouse

# Copy requirements.txt
COPY requirements.txt /app/requirements.txt 

# Install dependencies
RUN pip install --no-cache-dir --break-system-packages -r /app/requirements.txt
RUN pip freeze > /app/requirements.txt

# Set CUDA variables
RUN export CUDA_HOME=/usr/local/cuda
RUN export LIBRARY_PATH="${CUDA_HOME}/lib64"
RUN export C_INCLUDE_PATH="${CUDA_HOME}/include"

# Copy the application code
COPY src /app/src

# Set up an entry point for training
ENTRYPOINT ["python", "/app/src/train.py"]