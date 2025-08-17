# Pytorch CUDA base image 
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime
 
# --- Python defaults ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Streamlit defaults (compose/.env can still override)
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    CSV_PATH=parsed_1000_ml_jobs_us.csv


# Minimal OS deps. build-essential helps if a wheel needs a tiny native build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# App working directory 
WORKDIR /app


# Copy only requirements first to maximize layer cache 
COPY requirements.txt .

# Install the runtime dependencies 
RUN pip install --no-cache-dir -r requirements.txt 

# Copy the rest of the app code (filtered by .dockerignore)
COPY . .

# Expose Streamlit's port 
EXPOSE 8501

# Start Streamlit. Address 0.0.0.0 so Docker can publish the port. 
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.runOnSave=true"]
