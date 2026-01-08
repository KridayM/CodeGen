FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy dependencies first (cache-friendly)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your single application file
COPY agent.py .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI using agent.py
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8000"]
