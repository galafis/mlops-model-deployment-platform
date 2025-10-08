"""
MLOps Model Deployment Platform
Author: Gabriel Demetrios Lafis
Year: 2025

Sistema para gerenciamento e deployment de modelos de ML em produção.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import json


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


class Model:
    """Representa um modelo de ML"""
    
    def __init__(self, metadata: ModelMetadata):
        self.metadata = metadata
        self.status = ModelStatus.TRAINING
        self.endpoint: Optional[str] = None
        self.deployment_history: List[Dict[str, Any]] = []
    
    def promote_to_staging(self) -> bool:
        """Promove modelo para staging"""
        if self.status == ModelStatus.TRAINING:
            self.status = ModelStatus.STAGED
            self._log_status_change("staged")
            return True
        return False
    
    def promote_to_production(self) -> bool:
        """Promove modelo para produção"""
        if self.status == ModelStatus.STAGED:
            self.status = ModelStatus.PRODUCTION
            self._log_status_change("production")
            return True
        return False
    
    def _log_status_change(self, new_status: str):
        """Registra mudança de status"""
        self.deployment_history.append({
            "timestamp": datetime.now(),
            "status": new_status,
            "version": self.metadata.version
        })


class ModelRegistry:
    """Registro centralizado de modelos"""
    
    def __init__(self):
        self.models: Dict[str, List[Model]] = {}
    
    def register_model(self, model: Model) -> bool:
        """Registra um novo modelo"""
        if model.metadata.name not in self.models:
            self.models[model.metadata.name] = []
        self.models[model.metadata.name].append(model)
        print(f"✓ Modelo '{model.metadata.name}' v{model.metadata.version} registrado")
        return True
    
    def get_latest_version(self, model_name: str) -> Optional[Model]:
        """Retorna a versão mais recente de um modelo"""
        if model_name in self.models and self.models[model_name]:
            return self.models[model_name][-1]
        return None

    def get_production_model(self, model_name: str) -> Optional[Model]:
        """Retorna o modelo em produção"""
        if model_name in self.models:
            for model in reversed(self.models[model_name]):
                if model.status == ModelStatus.PRODUCTION:
                    return model
        return None


class DeploymentPlatform:
    """Plataforma de deployment de modelos"""
    
    def __init__(self, name: str):
        self.name = name
        self.registry = ModelRegistry()
        self.deployments: Dict[str, Dict[str, Any]] = {}
    
    def deploy_model(
        self,
        model: Model,
        config: DeploymentConfig
    ) -> bool:
        """Deploy de um modelo"""
        deployment_id = f"{model.metadata.name}-{model.metadata.version}"
        
        # Simular deployment
        self.deployments[deployment_id] = {
            "model": model,
            "config": config,
            "deployed_at": datetime.now(),
            "endpoint": f"https://api.ml-platform.com/v1/models/{deployment_id}/predict",
            "status": "running"
        }
        
        model.endpoint = self.deployments[deployment_id]["endpoint"]
        
        print(f"✓ Modelo deployed com estratégia {config.strategy.value}")
        print(f"  Endpoint: {model.endpoint}")
        return True

    def predict(self, model_name: str, model_version: str, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Realiza uma previsão usando um modelo implantado."""
        deployment_id = f"{model_name}-{model_version}"
        if deployment_id not in self.deployments:
            print(f"✗ Erro: Modelo {model_name} v{model_version} não encontrado ou não implantado.")
            return None

    def list_models(self) -> List[ModelMetadata]:
        """Lista todos os modelos registrados na plataforma."""
        all_models = []
        for model_name in self.registry.models:
            for model in self.registry.models[model_name]:
                all_models.append(model.metadata)
        return all_models

    def list_deployments(self) -> List[str]:
        """Lista todos os IDs de deployments ativos."""
        return list(self.deployments.keys())

        # Simulação de previsão
        # Em um cenário real, aqui haveria a chamada ao modelo implantado
        print(f"Simulando previsão para modelo {model_name} v{model_version} com entrada: {input_data}")
        if "transaction_amount" in input_data and input_data["transaction_amount"] > 500:
            return {"is_fraud": True, "score": 0.95}
        return {"is_fraud": False, "score": 0.15}
    
    def get_deployment_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Retorna informações do deployment"""
        for dep_id, dep_info in self.deployments.items():
            if dep_info["model"].metadata.name == model_name:
                return {
                    "deployment_id": dep_id,
                    "endpoint": dep_info["endpoint"],
                    "status": dep_info["status"],
                    "deployed_at": dep_info["deployed_at"].isoformat()
                }
        return None

    def list_models(self) -> List[ModelMetadata]:
        """Lista todos os modelos registrados na plataforma."""
        all_models = []
        for model_name in self.registry.models:
            for model in self.registry.models[model_name]:
                all_models.append(model.metadata)
        return all_models

    def list_deployments(self) -> List[str]:
        """Lista todos os IDs de deployments ativos."""
        return list(self.deployments.keys())


def example_usage():
    """Exemplo de uso"""
    
    # Criar plataforma
    platform = DeploymentPlatform("production-platform")
    
    # Criar modelo
    metadata = ModelMetadata(
        name="customer-churn-predictor",
        version="1.0.0",
        framework="scikit-learn",
        author="data-science-team@company.com",
        description="Modelo para predição de churn de clientes",
        metrics={"accuracy": 0.92, "precision": 0.89, "recall": 0.91},
        tags=["classification", "churn", "production"]
    )
    
    model = Model(metadata)
    
    # Registrar modelo
    platform.registry.register_model(model)
    
    # Promover para staging
    print("\n📊 Promovendo para staging...")
    model.promote_to_staging()
    print(f"  Status: {model.status.value}")
    
    # Promover para produção
    print("\n🚀 Promovendo para produção...")
    model.promote_to_production()
    print(f"  Status: {model.status.value}")
    
    # Deploy
    print("\n📦 Realizando deployment...")
    config = DeploymentConfig(
        strategy=DeploymentStrategy.BLUE_GREEN,
        replicas=3,
        auto_scaling=True
    )
    platform.deploy_model(model, config)
    
    # Informações do deployment
    print("\n📋 Informações do deployment:")
    info = platform.get_deployment_info("customer-churn-predictor")
    if info:
        for key, value in info.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    print("=" * 80)
    print("MLOps Model Deployment Platform - Example")
    print("=" * 80)
    example_usage()
