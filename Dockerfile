# Use Python 3.12 (Stable for TensorFlow)
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install system dependencies for OpenCV/Pillow
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the port Flask runs on
EXPOSE 7860

# Start the app using Gunicorn on port 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
