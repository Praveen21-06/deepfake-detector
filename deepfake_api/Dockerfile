# Use official Python slim base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for torchvision/Pillow)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgcc1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy API code
COPY . .

# Expose port 8000 for Uvicorn
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
