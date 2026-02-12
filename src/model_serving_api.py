from flask import Flask, jsonify, request
from src.model_deployment import (
    DeploymentPlatform,
    ModelMetadata,
    Model,
    DeploymentConfig,
    DeploymentStrategy,
    ModelStatus,
)

app = Flask(__name__)
platform_api = DeploymentPlatform("MLOpsPlatformAPI")


@app.route("/predict/<string:model_name>/<string:version>", methods=["POST"])
def predict_endpoint(model_name, version):
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": 'Formato de entrada inválido. Esperado {"features": [[...]]}'}), 400

    model = platform_api.registry.get_model(model_name, version)
    if not model or model.status != ModelStatus.PRODUCTION:
        return jsonify({"error": f"Modelo {model_name} v{version} não encontrado ou não está em produção"}), 404

    try:
        prediction_result = model.predict(data)
        return jsonify(prediction_result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/models", methods=["GET"])
def list_models_endpoint():
    models_info = platform_api.registry.list_models()
    return jsonify(models_info), 200


@app.route("/deployments", methods=["GET"])
def list_deployments_endpoint():
    deployments_info = []
    for dep_id, dep_data in platform_api.deployments.items():
        deployments_info.append(
            {
                "id": dep_id,
                "model_name": dep_data["model"].metadata.name,
                "model_version": dep_data["model"].metadata.version,
                "status": dep_data["status"],
                "endpoint": dep_data["endpoint"],
                "strategy": dep_data["config"].strategy.value,
                "replicas": dep_data["config"].replicas,
                "canary_traffic_percentage": dep_data["config"].canary_traffic_percentage,
            }
        )
    return jsonify(deployments_info), 200


@app.route("/register_model", methods=["POST"])
def register_model_endpoint():
    data = request.get_json()
    required_fields = ["name", "version", "framework", "author", "description"]
    if not data or not all(field in data for field in required_fields):
        return jsonify({"error": f"Campos obrigatórios: {required_fields}"}), 400

    metadata = ModelMetadata(
        name=data["name"],
        version=data["version"],
        framework=data["framework"],
        author=data["author"],
        description=data["description"],
        metrics=data.get("metrics", {}),
        tags=data.get("tags", []),
        model_path=data.get("model_path"),
    )
    model = Model(metadata)

    if platform_api.registry.register_model(model):
        model.promote_to_staging()
        platform_api.registry._save_registry()
        return (
            jsonify(
                {
                    "status": "Model registered and promoted to STAGED",
                    "model_name": model.metadata.name,
                    "version": model.metadata.version,
                }
            ),
            201,
        )
    return jsonify({"error": "Failed to register model"}), 400


@app.route("/deploy_model", methods=["POST"])
def deploy_model_endpoint():
    data = request.get_json()
    if not data or "model_name" not in data or "model_version" not in data or "strategy" not in data:
        return jsonify({"error": "Campos obrigatórios: model_name, model_version, strategy"}), 400

    model = platform_api.registry.get_model(data["model_name"], data["model_version"])
    if not model:
        return jsonify({"error": f"Modelo {data['model_name']} v{data['model_version']} não encontrado"}), 404

    if model.status == ModelStatus.STAGED:
        model.promote_to_production()
        platform_api.registry._save_registry()

    strategy_str = data["strategy"].lower()
    config = DeploymentConfig(
        strategy=DeploymentStrategy(strategy_str),
        replicas=data.get("replicas", 1),
        auto_scaling=data.get("auto_scaling", True),
        canary_traffic_percentage=data.get("canary_traffic_percentage"),
    )

    endpoint = platform_api.deploy_model(model, config)
    if endpoint:
        return jsonify({"status": "Model deployed", "endpoint": endpoint}), 201
    return jsonify({"error": "Failed to deploy model"}), 400


@app.route("/undeploy_model", methods=["POST"])
def undeploy_model_endpoint():
    data = request.get_json()
    if not data or "model_name" not in data or "model_version" not in data:
        return jsonify({"error": "Campos obrigatórios: model_name, model_version"}), 400

    if platform_api.undeploy_model(data["model_name"], data["model_version"]):
        return jsonify({"status": "Model undeployed"}), 200
    return jsonify({"error": "Failed to undeploy model"}), 400


@app.route("/reload_platform", methods=["POST"])
def reload_platform_endpoint():
    global platform_api
    platform_api = DeploymentPlatform("MLOpsPlatformAPI")
    return jsonify({"status": "Plataforma de deployment recarregada com sucesso"}), 200


def main():
    """
    Entry point for running the API server from command line.
    """
    import argparse

    parser = argparse.ArgumentParser(description="MLOps Model Serving API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5001, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    print(f"Starting MLOps Model Serving API on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()
