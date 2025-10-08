
"""
MLOps Model Deployment Platform
Author: Gabriel Demetrios Lafis
Year: 2025

Sistema para gerenciamento e deployment de modelos de ML em produção.
Esta versão inclui versionamento de modelos, estratégias de deployment mais detalhadas,
monitoramento básico e uma API Flask para inferência.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import json
import os
import time

# Dependências opcionais (instalar com `pip install flask`)
try:
    from flask import Flask, jsonify, request
except ImportError:
    Flask = None


class ModelStatus(Enum):
    """Status do modelo"""
    TRAINING = "training"
    STAGED = "staged"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class DeploymentStrategy(Enum):
    """Estratégias de deployment"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    SHADOW = "shadow"


@dataclass
class ModelMetadata:
    """Metadados do modelo"""
    name: str
    version: str
    framework: str  # tensorflow, pytorch, sklearn, etc.
    author: str
    description: str
    created_at: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    # Adicionado para persistência
    model_path: Optional[str] = None

    def to_dict(self):
        return {
            "name": self.name,
            "version": self.version,
            "framework": self.framework,
            "author": self.author,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "metrics": self.metrics,
            "tags": self.tags,
            "model_path": self.model_path
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class DeploymentConfig:
    """Configuração de deployment"""
    strategy: DeploymentStrategy
    replicas: int = 1
    cpu_limit: str = "1000m"
    memory_limit: str = "2Gi"
    auto_scaling: bool = True
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    canary_traffic_percentage: Optional[int] = None # Para estratégia Canary

    def to_dict(self):
        return {
            "strategy": self.strategy.value,
            "replicas": self.replicas,
            "cpu_limit": self.cpu_limit,
            "memory_limit": self.memory_limit,
            "auto_scaling": self.auto_scaling,
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "target_cpu_utilization": self.target_cpu_utilization,
            "canary_traffic_percentage": self.canary_traffic_percentage
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        data["strategy"] = DeploymentStrategy(data["strategy"])
        data.setdefault("canary_traffic_percentage", None)
        return cls(**data)


class Model:
    """Representa um modelo de ML"""
    
    def __init__(self, metadata: ModelMetadata):
        self.metadata = metadata
        self.status = ModelStatus.TRAINING
        self.endpoint: Optional[str] = None
        self.deployment_history: List[Dict[str, Any]] = []
        # Simula o carregamento do modelo (ex: um objeto sklearn, um modelo TF/PyTorch)
        self._model_instance = self._load_model_instance(metadata.model_path)

    def _load_model_instance(self, model_path: Optional[str]):
        """
        Simula o carregamento de um modelo a partir de um caminho.
        Em um cenário real, isso carregaria o modelo real (ex: pickle, saved_model).
        """
        if model_path and os.path.exists(model_path):
            print(f"[INFO] Carregando modelo de: {model_path}")
            try:
                import pickle
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"[ERROR] Erro ao carregar o modelo de {model_path}: {e}")
                return "MockModelInstance"
        print("[INFO] Nenhum caminho de modelo fornecido ou arquivo não encontrado. Usando mock.")
        return "MockModelInstance"

    def promote_to_staging(self) -> bool:
        """Promove modelo para staging"""
        if self.status in [ModelStatus.TRAINING, ModelStatus.ARCHIVED]:
            self.status = ModelStatus.STAGED
            self._log_status_change("staged")
            print(f"✓ Modelo {self.metadata.name} v{self.metadata.version} promovido para STAGED.")
            return True
        print(f"✗ Não foi possível promover o modelo {self.metadata.name} v{self.metadata.version} para STAGED. Status atual: {self.status.value}")
        return False
    
    def promote_to_production(self) -> bool:
        """
        Promove modelo para produção.
        Requer que o modelo esteja em STAGED.
        """
        if self.status == ModelStatus.STAGED:
            self.status = ModelStatus.PRODUCTION
            self._log_status_change("production")
            print(f"✓ Modelo {self.metadata.name} v{self.metadata.version} promovido para PRODUCTION.")
            return True
        print(f"✗ Não foi possível promover o modelo {self.metadata.name} v{self.metadata.version} para PRODUCTION. Status atual: {self.status.value}. Deve estar em STAGED.")
        return False

    def archive_model(self) -> bool:
        """
        Arquiva o modelo, tornando-o indisponível para novos deployments.
        """
        if self.status != ModelStatus.ARCHIVED:
            self.status = ModelStatus.ARCHIVED
            self._log_status_change("archived")
            print(f"✓ Modelo {self.metadata.name} v{self.metadata.version} ARQUIVADO.")
            return True
        print(f"✗ Modelo {self.metadata.name} v{self.metadata.version} já está ARQUIVADO.")
        return False
    
    def _log_status_change(self, new_status: str):
        """Registra mudança de status"""
        self.deployment_history.append({
            "timestamp": datetime.now().isoformat(),
            "status": new_status,
            "version": self.metadata.version
        })

    def predict(self, input_data: Any) -> Dict[str, Any]:
        """
        Realiza uma previsão do modelo com base nos dados de entrada.
        Utiliza a instância do modelo carregada.
        """
        if self._model_instance == "MockModelInstance":
            # Lógica de simulação simples para o mock
            print(f"  Simulando previsão para modelo {self.metadata.name} v{self.metadata.version} com input: {input_data}")
            # Assume que input_data é um dicionário com uma chave 'features' contendo uma lista de listas
            if isinstance(input_data, dict) and "features" in input_data and input_data["features"]:
                # Apenas para simulação, verifica o primeiro elemento da primeira feature
                if input_data["features"][0][0] > 0.5:
                    return {"prediction": 1, "probability": 0.85, "model_version": self.metadata.version}
                else:
                    return {"prediction": 0, "probability": 0.15, "model_version": self.metadata.version}
            return {"prediction": 0, "probability": 0.5, "model_version": self.metadata.version, "note": "Mock prediction for unexpected input format"}
        else:
            # Em um cenário real, aqui seria a chamada ao modelo carregado
            # Ex: return self._model_instance.predict(input_data)
            # Para o exemplo avançado, esperamos um array ou lista de features
            if isinstance(input_data, dict) and "features" in input_data:
                try:
                    import numpy as np
                    # Converte a lista de listas para um array numpy
                    features_array = np.array(input_data["features"])
                    prediction = self._model_instance.predict(features_array).tolist()
                    if hasattr(self._model_instance, 'predict_proba'):
                        prediction_proba = self._model_instance.predict_proba(features_array).tolist()
                        return {"prediction": prediction, "probabilities": prediction_proba, "model_version": self.metadata.version}
                    else:
                        return {"prediction": prediction, "model_version": self.metadata.version}
                except Exception as e:
                    return {"error": f"Erro ao usar modelo real para previsão: {e}", "model_version": self.metadata.version}
            return {"error": "Formato de entrada inesperado para modelo real", "model_version": self.metadata.version}


class ModelRegistry:
    """Registro centralizado de modelos com persistência em arquivo JSON"""
    
    def __init__(self, registry_file: str = "model_registry.json"):
        self.registry_file = registry_file
        self.models: Dict[str, List[Model]] = self._load_registry()
    
    def _load_registry(self) -> Dict[str, List[Model]]:
        """
        Carrega o registro de modelos de um arquivo JSON.
        """
        if os.path.exists(self.registry_file):
            with open(self.registry_file, "r") as f:
                data = json.load(f)
                loaded_models = {}
                for model_name, versions_data in data.items():
                    loaded_models[model_name] = []
                    for version_data in versions_data:
                        metadata = ModelMetadata.from_dict(version_data["metadata"])
                        model = Model(metadata)
                        model.status = ModelStatus(version_data["status"])
                        model.endpoint = version_data["endpoint"]
                        model.deployment_history = version_data["deployment_history"]
                        # O _model_instance é carregado no __init__ do Model, mas precisamos garantir que o caminho esteja correto
                        # e que ele seja recarregado se o arquivo existir.
                        model._model_instance = model._load_model_instance(metadata.model_path)
                        loaded_models[model_name].append(model)
                return loaded_models
        return {}

    def _save_registry(self):
        """
        Salva o registro de modelos em um arquivo JSON.
        """
        data_to_save = {}
        for model_name, versions in self.models.items():
            data_to_save[model_name] = []
            for model in versions:
                data_to_save[model_name].append({
                    "metadata": model.metadata.to_dict(),
                    "status": model.status.value,
                    "endpoint": model.endpoint,
                    "deployment_history": model.deployment_history
                })
        with open(self.registry_file, "w") as f:
            json.dump(data_to_save, f, indent=2)
        print(f"✓ Registro de modelos salvo em {self.registry_file}")

    def register_model(self, model: Model) -> bool:
        """
        Registra um novo modelo ou uma nova versão de um modelo existente.
        """
        if model.metadata.name not in self.models:
            self.models[model.metadata.name] = []
        
        # Verificar se a versão já existe
        if any(m.metadata.version == model.metadata.version for m in self.models[model.metadata.name]):
            print(f"⚠ Modelo '{model.metadata.name}' v{model.metadata.version} já registrado. Use update_model para modificar.")
            return False

        self.models[model.metadata.name].append(model)
        self.models[model.metadata.name].sort(key=lambda m: m.metadata.created_at) # Manter ordenado por criação
        self._save_registry()
        print(f"✓ Modelo '{model.metadata.name}' v{model.metadata.version} registrado com sucesso.")
        return True
    
    def get_model(self, model_name: str, version: Optional[str] = None) -> Optional[Model]:
        """
        Retorna um modelo específico pela versão ou a versão mais recente se nenhuma for especificada.
        """
        if model_name not in self.models:
            return None
        
        if version:
            for model in self.models[model_name]:
                if model.metadata.version == version:
                    return model
        else:
            # Retorna a versão mais recente (última na lista ordenada)
            if self.models[model_name]:
                return self.models[model_name][-1]
        return None

    def get_production_model(self, model_name: str) -> Optional[Model]:
        """
        Retorna o modelo atualmente em produção para um dado nome.
        """
        if model_name in self.models:
            for model in reversed(self.models[model_name]): # Busca da mais recente para a mais antiga
                if model.status == ModelStatus.PRODUCTION:
                    return model
        return None

    def list_models(self) -> List[Dict[str, Any]]:
        """
        Lista todos os modelos e suas versões registradas.
        """
        all_models_info = []
        for model_name, versions in self.models.items():
            for model in versions:
                all_models_info.append({
                    "name": model.metadata.name,
                    "version": model.metadata.version,
                    "status": model.status.value,
                    "framework": model.metadata.framework,
                    "endpoint": model.endpoint
                })
        return all_models_info


class DeploymentPlatform:
    """Plataforma de deployment de modelos"""
    
    def __init__(self, name: str, registry_file: str = "model_registry.json"):
        self.name = name
        self.registry = ModelRegistry(registry_file)
        self.deployments: Dict[str, Dict[str, Any]] = self._load_deployments()

    def _load_deployments(self) -> Dict[str, Dict[str, Any]]:
        """
        Carrega os deployments ativos de um arquivo JSON.
        """
        deployments_file = "model_deployments.json"
        if os.path.exists(deployments_file):
            with open(deployments_file, "r") as f:
                data = json.load(f)
                loaded_deployments = {}
                for dep_id, dep_data in data.items():
                    model = self.registry.get_model(dep_data["model_name"], dep_data["model_version"])
                    if model:
                        # Garante que o _model_instance seja carregado no modelo recuperado do registro
                        model._model_instance = model._load_model_instance(model.metadata.model_path)
                        config = DeploymentConfig.from_dict(dep_data["config"])
                        loaded_deployments[dep_id] = {
                            "model": model,
                            "config": config,
                            "deployed_at": datetime.fromisoformat(dep_data["deployed_at"]),
                            "endpoint": dep_data["endpoint"],
                            "status": dep_data["status"]
                        }
                    else:
                        print(f"[WARN] Modelo {dep_data['model_name']} v{dep_data['model_version']} não encontrado no registro para deployment {dep_id}.")
                return loaded_deployments
        return {}

    def _save_deployments(self):
        """
        Salva os deployments ativos em um arquivo JSON.
        """
        deployments_file = "model_deployments.json"
        data_to_save = {}
        for dep_id, dep_info in self.deployments.items():
            data_to_save[dep_id] = {
                "model_name": dep_info["model"].metadata.name,
                "model_version": dep_info["model"].metadata.version,
                "config": dep_info["config"].to_dict(),
                "deployed_at": dep_info["deployed_at"].isoformat(),
                "endpoint": dep_info["endpoint"],
                "status": dep_info["status"]
            }
        with open(deployments_file, "w") as f:
            json.dump(data_to_save, f, indent=2)
        print(f"✓ Deployments salvos em {deployments_file}")

    def deploy_model(
        self,
        model: Model,
        config: DeploymentConfig
    ) -> Optional[str]:
        """
        Realiza o deployment de um modelo com uma estratégia específica.
        Retorna o endpoint do modelo ou None em caso de falha.
        """
        if model.status not in [ModelStatus.STAGED, ModelStatus.PRODUCTION]:
            print(f"✗ Erro: Modelo {model.metadata.name} v{model.metadata.version} não está em STAGED ou PRODUCTION. Status atual: {model.status.value}")
            return None

        deployment_id = f"{model.metadata.name}-{model.metadata.version}"
        if deployment_id in self.deployments:
            print(f"✗ Erro: Modelo {model.metadata.name} v{model.metadata.version} já possui um deployment ativo.")
            return None

        # Simula a criação de um endpoint
        endpoint = f"http://127.0.0.1:5001/predict/{model.metadata.name}/{model.metadata.version}"
        model.endpoint = endpoint

        self.deployments[deployment_id] = {
            "model": model,
            "config": config,
            "deployed_at": datetime.now(),
            "endpoint": endpoint,
            "status": "active"
        }
        self._save_deployments()
        print(f"✓ Modelo {model.metadata.name} v{model.metadata.version} implantado com sucesso. Estratégia: {config.strategy.value}")
        return endpoint

    def undeploy_model(self, model_name: str, version: str) -> bool:
        """
        Remove um modelo do deployment ativo.
        """
        deployment_id = f"{model_name}-{version}"
        if deployment_id in self.deployments:
            del self.deployments[deployment_id]
            self._save_deployments()
            print(f"✓ Modelo {model_name} v{version} removido do deployment.")
            return True
        print(f"✗ Deployment para o modelo {model_name} v{version} não encontrado.")
        return False

    def promote_canary_to_production(self, model: Model) -> bool:
        """
        Promove a versão canary para 100% do tráfego, tornando-a a nova versão de produção.
        Assume que o modelo já está em um deployment canary ativo.
        """
        deployment_id = f"{model.metadata.name}-{model.metadata.version}"
        if deployment_id not in self.deployments:
            print(f"✗ Erro: Deployment canary para {model.metadata.name} v{model.metadata.version} não encontrado.")
            return False

        current_deployment = self.deployments[deployment_id]
        if current_deployment["config"].strategy != DeploymentStrategy.CANARY:
            print(f"✗ Erro: O deployment {deployment_id} não é uma estratégia Canary ativa.")
            return False

        print(f"  Promovendo canary {model.metadata.name} v{model.metadata.version} para produção completa...")
        # Simula a atualização da configuração de tráfego para 100%
        current_deployment["config"].canary_traffic_percentage = 100
        current_deployment["config"].strategy = DeploymentStrategy.BLUE_GREEN # Ou PRODUCTION_ROLLOUT
        model.promote_to_production() # Promove o modelo no registro para PRODUCTION
        self.registry._save_registry() # Salva o registro após a promoção do modelo
        self._save_deployments()
        print(f"✓ Canary {model.metadata.name} v{model.metadata.version} promovido para produção completa.")
        return True

    def scale_deployment(self, model_name: str, version: str, new_replicas: int) -> bool:
        """
        Escala o número de réplicas para um deployment específico.
        """
        deployment_id = f"{model_name}-{version}"
        if deployment_id in self.deployments:
            self.deployments[deployment_id]["config"].replicas = new_replicas
            self._save_deployments()
            print(f"✓ Deployment {model_name} v{version} escalado para {new_replicas} réplicas.")
            return True
        print(f"✗ Deployment {deployment_id} não encontrado.")
        return False


# --- API de Serviço de Modelos (Flask) ---

if Flask:
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
        platform_api = DeploymentPlatform("MLOpsPlatformAPI") # Recarrega a instância da plataforma
        return jsonify({"status": "Plataforma de deployment recarregada com sucesso"}), 200

    if __name__ == "__main__":
        app.run(port=5001, debug=False, use_reloader=False)

