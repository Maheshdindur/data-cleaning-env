FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose the port the app runs on
EXPOSE 7860

# Command to run the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]