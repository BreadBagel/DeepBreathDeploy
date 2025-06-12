# 1. Base image
FROM python:3.11-slim

# 2. Prevent Python from writing .pyc files and enable stdout/stderr unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set working directory in the container
WORKDIR /app

# 4. Copy only requirements first (better cache)
COPY requirementsupdated.txt requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 5. Copy your application code
#    We include top-level scripts, folders, and model files
COPY \
    *.py \
    config.py \
    api/ \
    routes/ \
    utils/ \
    restnet_old_pth/ \
    pkl\ folder/ \
    models.py \
    *.pth \
    *.pkl \
    ./

# 6. (Optional) Expose ports if your app listens on one, e.g.:
#    EXPOSE 5000

# 7. Default command (adjust entrypoint script as needed)
CMD ["python", "app.py"]
