FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir numpy iterative-stratification

COPY build_encoders.py .
CMD ["python", "build_encoders.py"]
