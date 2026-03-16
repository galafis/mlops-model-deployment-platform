FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001

CMD ["python", "-c", "from src.model_deployment import DeploymentPlatform; platform = DeploymentPlatform('docker-platform'); app = platform.create_flask_api(); app.run(host='0.0.0.0', port=5001)"]
