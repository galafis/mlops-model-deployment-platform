# API Documentation

## REST API Endpoints

The MLOps Model Deployment Platform provides a RESTful API for model management, deployment, and inference.

### Base URL

```
http://localhost:5001
```

---

## Endpoints

### 1. Model Management

#### List All Models

Get a list of all registered models.

**Endpoint:** `GET /models`

**Response:**
```json
[
  {
    "name": "model-name",
    "version": "1.0.0",
    "status": "production",
    "framework": "scikit-learn",
    "endpoint": "http://127.0.0.1:5001/predict/model-name/1.0.0"
  }
]
```

**Example:**
```bash
curl http://localhost:5001/models
```

---

#### Register Model

Register a new model or a new version of an existing model.

**Endpoint:** `POST /register_model`

**Request Body:**
```json
{
  "name": "my-classifier",
  "version": "1.0.0",
  "framework": "scikit-learn",
  "author": "user@example.com",
  "description": "Classification model for customer churn",
  "metrics": {
    "accuracy": 0.95,
    "f1_score": 0.93
  },
  "tags": ["classification", "production"],
  "model_path": "./models/classifier_v1.pkl"
}
```

**Response:**
```json
{
  "status": "Model registered and promoted to STAGED",
  "model_name": "my-classifier",
  "version": "1.0.0"
}
```

**Example:**
```bash
curl -X POST http://localhost:5001/register_model \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-classifier",
    "version": "1.0.0",
    "framework": "scikit-learn",
    "author": "user@example.com",
    "description": "My classification model"
  }'
```

---

### 2. Deployment Management

#### List All Deployments

Get a list of all active deployments.

**Endpoint:** `GET /deployments`

**Response:**
```json
[
  {
    "id": "model-name-1.0.0",
    "model_name": "model-name",
    "model_version": "1.0.0",
    "status": "running",
    "endpoint": "http://127.0.0.1:5001/predict/model-name/1.0.0",
    "strategy": "blue_green",
    "replicas": 3,
    "canary_traffic_percentage": null
  }
]
```

**Example:**
```bash
curl http://localhost:5001/deployments
```

---

#### Deploy Model

Deploy a model using a specific strategy.

**Endpoint:** `POST /deploy_model`

**Request Body:**
```json
{
  "model_name": "my-classifier",
  "model_version": "1.0.0",
  "strategy": "BLUE_GREEN",
  "replicas": 3,
  "auto_scaling": true,
  "min_replicas": 2,
  "max_replicas": 10,
  "canary_traffic_percentage": null
}
```

**Deployment Strategies:**
- `BLUE_GREEN` - Zero-downtime deployment with instant rollback
- `CANARY` - Gradual rollout with traffic percentage control
- `ROLLING` - Progressive instance replacement
- `SHADOW` - Mirror traffic without affecting production

**Response:**
```json
{
  "status": "Model deployed",
  "endpoint": "http://127.0.0.1:5001/predict/my-classifier/1.0.0"
}
```

**Example:**
```bash
curl -X POST http://localhost:5001/deploy_model \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "my-classifier",
    "model_version": "1.0.0",
    "strategy": "BLUE_GREEN",
    "replicas": 2
  }'
```

---

#### Undeploy Model

Remove a model from deployment.

**Endpoint:** `POST /undeploy_model`

**Request Body:**
```json
{
  "model_name": "my-classifier",
  "model_version": "1.0.0"
}
```

**Response:**
```json
{
  "status": "Model undeployed"
}
```

**Example:**
```bash
curl -X POST http://localhost:5001/undeploy_model \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "my-classifier",
    "model_version": "1.0.0"
  }'
```

---

### 3. Inference

#### Make Prediction

Get predictions from a deployed model.

**Endpoint:** `POST /predict/{model_name}/{version}`

**Request Body:**
```json
{
  "features": [[0.5, 0.3, 0.8, 0.2]]
}
```

**Response:**
```json
{
  "prediction": [1],
  "probabilities": [[0.2, 0.8]],
  "model_version": "1.0.0"
}
```

**Example:**
```bash
curl -X POST http://localhost:5001/predict/my-classifier/1.0.0 \
  -H "Content-Type: application/json" \
  -d '{"features": [[0.5, 0.3, 0.8, 0.2]]}'
```

---

### 4. Platform Management

#### Reload Platform

Reload the platform configuration and state from disk.

**Endpoint:** `POST /reload_platform`

**Response:**
```json
{
  "status": "Plataforma de deployment recarregada com sucesso"
}
```

**Example:**
```bash
curl -X POST http://localhost:5001/reload_platform
```

---

## Python API

### DeploymentPlatform

Main class for managing model deployments.

#### Initialization

```python
from src.model_deployment import DeploymentPlatform

platform = DeploymentPlatform(
    name="my-platform",
    registry_file="model_registry.json"
)
```

#### Methods

##### `register_model(model: Model) -> bool`

Register a model in the platform.

```python
from src.model_deployment import Model, ModelMetadata

metadata = ModelMetadata(
    name="classifier",
    version="1.0.0",
    framework="scikit-learn",
    author="user@example.com",
    description="My model"
)
model = Model(metadata)
platform.registry.register_model(model)
```

##### `deploy_model(model: Model, config: DeploymentConfig) -> str`

Deploy a model with specified configuration.

```python
from src.model_deployment import DeploymentConfig, DeploymentStrategy

config = DeploymentConfig(
    strategy=DeploymentStrategy.BLUE_GREEN,
    replicas=3
)
endpoint = platform.deploy_model(model, config)
```

##### `predict(model_name: str, version: str, input_data: dict) -> dict`

Make a prediction with a deployed model.

```python
prediction = platform.predict(
    "classifier",
    "1.0.0",
    {"features": [[0.5, 0.3, 0.8]]}
)
```

##### `scale_deployment(model_name: str, version: str, new_replicas: int) -> bool`

Scale a deployment to a new number of replicas.

```python
platform.scale_deployment("classifier", "1.0.0", 5)
```

##### `undeploy_model(model_name: str, version: str) -> bool`

Remove a model from deployment.

```python
platform.undeploy_model("classifier", "1.0.0")
```

---

### Model

Class representing a machine learning model.

#### Methods

##### `promote_to_staging() -> bool`

Promote model to staging environment.

```python
model.promote_to_staging()
```

##### `promote_to_production() -> bool`

Promote model to production environment.

```python
model.promote_to_production()
```

##### `archive_model() -> bool`

Archive a model, making it unavailable for deployment.

```python
model.archive_model()
```

##### `predict(input_data: Any) -> dict`

Make a prediction with the model.

```python
result = model.predict({"features": [[0.5, 0.3, 0.8]]})
```

---

### ModelRegistry

Centralized registry for model management.

#### Methods

##### `get_model(model_name: str, version: str = None) -> Model`

Get a specific model or the latest version.

```python
model = platform.registry.get_model("classifier", "1.0.0")
latest = platform.registry.get_model("classifier")  # Gets latest version
```

##### `get_production_model(model_name: str) -> Model`

Get the model currently in production.

```python
prod_model = platform.registry.get_production_model("classifier")
```

##### `list_models() -> List[dict]`

List all registered models.

```python
models = platform.registry.list_models()
for m in models:
    print(f"{m['name']} v{m['version']} - {m['status']}")
```

---

## Error Handling

### HTTP Status Codes

- `200` - Success
- `201` - Created (successful registration/deployment)
- `400` - Bad Request (invalid input)
- `404` - Not Found (model not found)
- `500` - Internal Server Error

### Error Response Format

```json
{
  "error": "Error message describing what went wrong"
}
```

---

## Rate Limiting

Currently no rate limiting is implemented. For production use, consider implementing:
- Request throttling
- Authentication/Authorization
- API keys
- Usage quotas

---

## Security Considerations

For production deployments:

1. **Authentication**: Implement JWT or OAuth2
2. **HTTPS**: Use TLS/SSL encryption
3. **Input Validation**: Validate all input data
4. **Rate Limiting**: Prevent abuse
5. **Logging**: Log all API access
6. **Monitoring**: Monitor API performance and errors

---

## Examples

See the [examples](examples/) directory for complete working examples:

- `simple_example.py` - Basic usage
- `../src/advanced_example.py` - Full MLOps workflow

---

## Support

For questions and issues:
- Check the main [README.md](README.md)
- Review [CONTRIBUTING.md](CONTRIBUTING.md)
- Open an issue on GitHub
