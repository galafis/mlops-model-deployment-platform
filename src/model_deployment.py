
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

    def to_dict(self):
        return {
            "strategy": self.strategy.value,
            "replicas": self.replicas,
            "cpu_limit": self.cpu_limit,
            "memory_limit": self.memory_limit,
            "auto_scaling": self.auto_scaling,
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "target_cpu_utilization": self.target_cpu_utilization
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        data["strategy"] = DeploymentStrategy(data["strategy"])
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
            # Exemplo: return pickle.load(open(model_path, 'rb'))
            return f"MockModelInstance({model_path})"
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

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Realiza uma previsão do modelo com base nos dados de entrada.
        Utiliza a instância do modelo carregada.
        """
        if self._model_instance == "MockModelInstance":
            # Lógica de simulação simples para o mock
            print(f"  Simulando previsão para modelo {self.metadata.name} v{self.metadata.version} com input: {input_data}")
            if "feature_1" in input_data and input_data["feature_1"] > 0.5:
                return {"prediction": 1, "probability": 0.85, "model_version": self.metadata.version}
            else:
                return {"prediction": 0, "probability": 0.15, "model_version": self.metadata.version}
        else:
            # Em um cenário real, aqui seria a chamada ao modelo carregado
            # Ex: return self._model_instance.predict(input_data)
            print(f"[INFO] Realizando previsão com modelo real {self._model_instance} para input: {input_data}")
            return {"prediction": "real_prediction", "confidence": 0.95, "model_version": self.metadata.version}


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
            print(f"⚠ Modelo \'{model.metadata.name}\' v{model.metadata.version} já registrado. Use update_model para modificar.")
            return False

        self.models[model.metadata.name].append(model)
        self.models[model.metadata.name].sort(key=lambda m: m.metadata.created_at) # Manter ordenado por criação
        self._save_registry()
        print(f"✓ Modelo \'{model.metadata.name}\' v{model.metadata.version} registrado com sucesso.")
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
                        config = DeploymentConfig.from_dict(dep_data["config"])
                        loaded_deployments[dep_id] = {
                            "model": model,
                            "config": config,
                            "deployed_at": datetime.fromisoformat(dep_data["deployed_at"]),
                            "endpoint": dep_data["endpoint"],
                            "status": dep_data["status"]
                        }
                    else:
                        print(f"[WARN] Modelo {dep_data["model_name"]} v{dep_data["model_version"]} não encontrado no registro para deployment {dep_id}.")
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
        
        # Simular diferentes estratégias de deployment
        print(f"[INFO] Iniciando deployment para {deployment_id} com estratégia {config.strategy.value}...")
        if config.strategy == DeploymentStrategy.BLUE_GREEN:
            print("  Implementando Blue/Green: Preparando nova infraestrutura...")
            time.sleep(1) # Simula tempo de provisionamento
            print("  Trocando tráfego para a nova versão (Green)...")
        elif config.strategy == DeploymentStrategy.CANARY:
            print("  Implementando Canary: Redirecionando pequena porcentagem de tráfego...")
            time.sleep(0.5)
            print("  Monitorando performance da versão Canary...")
        elif config.strategy == DeploymentStrategy.ROLLING:
            print("  Implementando Rolling Update: Atualizando instâncias gradualmente...")
            time.sleep(0.5)
        elif config.strategy == DeploymentStrategy.SHADOW:
            print("  Implementando Shadow: Enviando tráfego para nova versão sem afetar usuários...")
            time.sleep(0.5)

        endpoint = f"https://api.ml-platform.com/v1/models/{model.metadata.name}/{model.metadata.version}/predict"
        self.deployments[deployment_id] = {
            "model": model,
            "config": config,
            "deployed_at": datetime.now(),
            "endpoint": endpoint,
            "status": "running"
        }
        model.endpoint = endpoint
        model.status = ModelStatus.PRODUCTION # Assume que o deployment bem-sucedido o coloca em produção
        self.registry._save_registry() # Salva o status atualizado do modelo
        self._save_deployments()
        
        print(f"✓ Modelo {deployment_id} deployed com sucesso. Endpoint: {endpoint}")
        return endpoint

    def undeploy_model(self, model_name: str, model_version: str) -> bool:
        """
        Remove um modelo implantado da plataforma.
        """
        deployment_id = f"{model_name}-{model_version}"
        if deployment_id in self.deployments:
            del self.deployments[deployment_id]
            model = self.registry.get_model(model_name, model_version)
            if model:
                model.endpoint = None
                model.status = ModelStatus.ARCHIVED # Após undeploy, o modelo é arquivado
                self.registry._save_registry()
            self._save_deployments()
            print(f"✓ Modelo {model_name} v{model_version} desimplantado com sucesso.")
            return True
        print(f"⚠ Modelo {model_name} v{model_version} não encontrado ou não implantado.")
        return False

    def scale_deployment(self, model_name: str, model_version: str, new_replicas: int) -> bool:
        """
        Ajusta o número de réplicas para um deployment existente.
        """
        deployment_id = f"{model_name}-{model_version}"
        if deployment_id in self.deployments:
            self.deployments[deployment_id]["config"].replicas = new_replicas
            self._save_deployments()
            print(f"✓ Deployment {deployment_id} escalado para {new_replicas} réplicas.")
            return True
        print(f"⚠ Deployment {deployment_id} não encontrado.")
        return False

    def predict(self, model_name: str, version: str, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Realiza uma previsão usando um modelo implantado.
        """
        deployment_id = f"{model_name}-{version}"
        if deployment_id not in self.deployments:
            print(f"✗ Erro: Modelo {model_name} v{version} não encontrado ou não implantado.")
            return None
        
        model = self.deployments[deployment_id]["model"]
        # Aqui, em um cenário real, haveria uma chamada HTTP para o endpoint do modelo
        # return requests.post(model.endpoint, json=input_data).json()
        return model.predict(input_data)

    def list_deployments(self) -> List[Dict[str, Any]]:
        """
        Lista todos os deployments ativos com suas informações.
        """
        deployments_info = []
        for dep_id, dep_info in self.deployments.items():
            deployments_info.append({
                "deployment_id": dep_id,
                "model_name": dep_info["model"].metadata.name,
                "model_version": dep_info["model"].metadata.version,
                "status": dep_info["status"],
                "endpoint": dep_info["endpoint"],
                "strategy": dep_info["config"].strategy.value,
                "replicas": dep_info["config"].replicas,
                "deployed_at": dep_info["deployed_at"].isoformat()
            })
        return deployments_info

    def create_flask_api(self):
        """
        Cria e retorna uma instância da aplicação Flask para a API REST de inferência.
        """
        if not Flask:
            raise ImportError("Flask não está instalado. Execute `pip install Flask`.")

        app = Flask(__name__)

        @app.route("/predict/<model_name>/<version>", methods=["POST"])
        def predict_endpoint(model_name, version):
            input_data = request.json
            if not input_data:
                return jsonify({"error": "Dados de entrada não fornecidos"}), 400
            
            prediction = self.predict(model_name, version, input_data)
            if prediction:
                return jsonify(prediction)
            return jsonify({"error": "Modelo não encontrado ou não implantado"}), 404

        @app.route("/models", methods=["GET"])
        def list_registered_models():
            models = self.registry.list_models()
            return jsonify(models)

        @app.route("/deployments", methods=["GET"])
        def list_active_deployments():
            deployments = self.list_deployments()
            return jsonify(deployments)

        return app


def example_usage():
    """
    Exemplo de uso da plataforma de deployment de modelos.
    """
    # Limpar arquivos de registro e deployment para um exemplo limpo
    if os.path.exists("model_registry.json"):
        os.remove("model_registry.json")
    if os.path.exists("model_deployments.json"):
        os.remove("model_deployments.json")

    platform = DeploymentPlatform("production-platform")

    # --- Modelo 1: Churn Predictor v1.0.0 ---
    print("\n" + "=" * 20 + " Churn Predictor v1.0.0 " + "=" * 20)
    metadata_v1 = ModelMetadata(
        name="customer-churn-predictor",
        version="1.0.0",
        framework="scikit-learn",
        author="data-science-team",
        description="Modelo para predição de churn de clientes",
        metrics={"accuracy": 0.92, "precision": 0.89},
        tags=["classification", "churn"],
        model_path="./models/churn_v1.pkl" # Caminho simulado
    )
    model_v1 = Model(metadata_v1)
    platform.registry.register_model(model_v1)
    model_v1.promote_to_staging()
    platform.deploy_model(model_v1, DeploymentConfig(strategy=DeploymentStrategy.BLUE_GREEN, replicas=2))

    # --- Modelo 2: Churn Predictor v1.1.0 (nova versão) ---
    print("\n" + "=" * 20 + " Churn Predictor v1.1.0 " + "=" * 20)
    metadata_v1_1 = ModelMetadata(
        name="customer-churn-predictor",
        version="1.1.0",
        framework="tensorflow",
        author="data-science-team",
        description="Modelo de churn aprimorado com rede neural",
        metrics={"accuracy": 0.94, "precision": 0.91},
        tags=["classification", "churn", "neural-network"],
        model_path="./models/churn_v1_1.h5" # Caminho simulado
    )
    model_v1_1 = Model(metadata_v1_1)
    platform.registry.register_model(model_v1_1)
    model_v1_1.promote_to_staging()
    platform.deploy_model(model_v1_1, DeploymentConfig(strategy=DeploymentStrategy.CANARY, replicas=1))

    # --- Modelo 3: Fraud Detector v2.0.0 ---
    print("\n" + "=" * 20 + " Fraud Detector v2.0.0 " + "=" * 20)
    metadata_fraud = ModelMetadata(
        name="fraud-detector",
        version="2.0.0",
        framework="pytorch",
        author="fraud-prevention-team",
        description="Modelo para detecção de fraude em transações",
        metrics={"f1_score": 0.90, "recall": 0.95},
        tags=["anomaly-detection", "fraud"],
        model_path="./models/fraud_v2.pt" # Caminho simulado
    )
    model_fraud = Model(metadata_fraud)
    platform.registry.register_model(model_fraud)
    model_fraud.promote_to_staging()
    platform.deploy_model(model_fraud, DeploymentConfig(strategy=DeploymentStrategy.ROLLING, replicas=3))

    print("\n" + "=" * 20 + " Listando Modelos e Deployments " + "=" * 20)
    print("\nModelos Registrados:")
    print(json.dumps(platform.registry.list_models(), indent=2))

    print("\nDeployments Ativos:")
    print(json.dumps(platform.list_deployments(), indent=2))

    print("\n" + "=" * 20 + " Realizando Previsões " + "=" * 20)
    input_data_churn = {"feature_1": 0.7, "feature_2": 10, "feature_3": "A"}
    input_data_fraud = {"transaction_amount": 1500.0, "transaction_type": "online"}

    print("\nPrevisão com Churn Predictor v1.0.0:")
    print(platform.predict("customer-churn-predictor", "1.0.0", input_data_churn))

    print("\nPrevisão com Churn Predictor v1.1.0:")
    print(platform.predict("customer-churn-predictor", "1.1.0", input_data_churn))

    print("\nPrevisão com Fraud Detector v2.0.0:")
    # O modelo de fraude usará o mock_model_instance, então o input_data_fraud será ignorado
    print(platform.predict("fraud-detector", "2.0.0", input_data_fraud))

    print("\n" + "=" * 20 + " Gerenciamento de Deployments " + "=" * 20)
    print("\nEscalando Churn Predictor v1.0.0 para 5 réplicas...")
    platform.scale_deployment("customer-churn-predictor", "1.0.0", 5)
    print("Deployments Ativos após escala:")
    print(json.dumps(platform.list_deployments(), indent=2))

    print("\nDesimplantando Churn Predictor v1.0.0...")
    platform.undeploy_model("customer-churn-predictor", "1.0.0")
    print("Deployments Ativos após desimplantação:")
    print(json.dumps(platform.list_deployments(), indent=2))

    print("\n" + "=" * 80)
    print("Exemplo Concluído")
    print("=" * 80)


if __name__ == "__main__":
    example_usage()

