# Use the official Python 3.11 image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install the uv package manager
RUN pip install uv

# Copy the project files into the container
COPY . /app

# Install dependencies using uv
RUN uv sync --no-dev

# Hugging Face Spaces requires applications to run on port 7860
EXPOSE 7860

# Command to run the FastAPI application
CMD ["uv", "run", "uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
