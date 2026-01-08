FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy dependencies first (cache-friendly)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI using main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
