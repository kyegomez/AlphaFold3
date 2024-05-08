# ==================================
# Use an official Python runtime as a parent image
FROM python:3.10-slim
RUN apt-get update && apt-get -y install libgl1-mesa-dev libglib2.0-0 build-esse
ntial; apt-get clean
RUN pip install opencv-contrib-python-headless

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /usr/src/zeta


# Install Python dependencies
# COPY requirements.txt and pyproject.toml if you're using poetry for dependency management
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir zetascale

# Copy the rest of the application
COPY . .

