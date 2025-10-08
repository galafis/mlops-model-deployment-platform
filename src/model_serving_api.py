from flask import Flask, jsonify, request
from src.model_deployment import DeploymentPlatform, ModelMetadata, Model, DeploymentConfig, DeploymentStrategy, ModelStatus
import threading
import time

app = Flask(__name__)
platform_api = DeploymentPlatform("MLOpsPlatformAPI") # Instância da plataforma para a API

@app.route("/predict/<string:model_name>/<string:version>", methods=["POST"])
def predict_endpoint(model_name, version):
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Formato de entrada inválido. Esperado {\"features\": [[...]]}"}), 400

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
        deployments_info.append({
            "id": dep_id,
            "model_name": dep_data["model"].metadata.name,
            "model_version": dep_data["model"].metadata.version,
            "status": dep_data["status"],
            "endpoint": dep_data["endpoint"],
            "strategy": dep_data["config"].strategy.value,
            "replicas": dep_data["config"].replicas,
            "canary_traffic_percentage": dep_data["config"].canary_traffic_percentage
        })
    return jsonify(deployments_info), 200

@app.route("/reload_platform", methods=["POST"])
def reload_platform_endpoint():
    global platform_api
    # Recarrega a instância da plataforma, o que por sua vez recarrega o registro e os deployments
    platform_api = DeploymentPlatform("MLOpsPlatformAPI")
    return jsonify({"status": "Plataforma de deployment recarregada com sucesso"}), 200

if __name__ == "__main__":
    app.run(port=5001, debug=False, use_reloader=False)

