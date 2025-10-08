import unittest
import sys
import os
from datetime import datetime
import json
import shutil

# Adicionar o diretório src ao path para importar os módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_deployment import (
    DeploymentPlatform,
    Model,
    ModelMetadata,
    ModelStatus,
    DeploymentStrategy,
    DeploymentConfig,
    ModelRegistry
)

# Mock Flask para testes de API
class MockFlaskClient:
    def __init__(self, app):
        self.app = app

    def get(self, path):
        with self.app.test_request_context(path):
            response = self.app.dispatch_request()
            return MockResponse(response)

    def post(self, path, json=None):
        with self.app.test_request_context(path, method='POST', json=json):
            response = self.app.dispatch_request()
            return MockResponse(response)

class MockResponse:
    def __init__(self, response):
        self.response = response
        self.status_code = response.status_code
        self.data = response.get_data()


class TestMLOpsIntegration(unittest.TestCase):
    def setUp(self):
        self.test_registry_file = "test_model_registry_integration.json"
        self.test_deployments_file = "test_model_deployments_integration.json"

        # Limpar arquivos de teste antes de cada execução
        if os.path.exists(self.test_registry_file):
            os.remove(self.test_registry_file)
        if os.path.exists(self.test_deployments_file):
            os.remove(self.test_deployments_file)

        self.platform = DeploymentPlatform(name="test-platform-integration", registry_file=self.test_registry_file)

        self.metadata_v1 = ModelMetadata(
            name="integration-model",
            version="1.0.0",
            framework="scikit-learn",
            author="test-author",
            description="A test integration model v1",
            model_path="./models/integration_test_v1.pkl"
        )
        self.model_v1 = Model(self.metadata_v1)

        self.metadata_v2 = ModelMetadata(
            name="integration-model",
            version="2.0.0",
            framework="tensorflow",
            author="test-author",
            description="A test integration model v2",
            model_path="./models/integration_test_v2.h5"
        )
        self.model_v2 = Model(self.metadata_v2)

        # Criar a aplicação Flask
        self.app = self.platform.create_flask_api()
        self.app.testing = True
        self.client = MockFlaskClient(self.app)

    def tearDown(self):
        # Limpar arquivos de teste após cada execução
        if os.path.exists(self.test_registry_file):
            os.remove(self.test_registry_file)
        if os.path.exists(self.test_deployments_file):
            os.remove(self.test_deployments_file)

    def test_full_deployment_lifecycle_via_api(self):
        """Testa o ciclo de vida completo de um modelo via API: registro, deploy, predição, undeploy"""
        # 1. Registrar modelo v1
        register_payload_v1 = {
            "name": self.model_v1.metadata.name,
            "version": self.model_v1.metadata.version,
            "framework": self.model_v1.metadata.framework,
            "author": self.model_v1.metadata.author,
            "description": self.model_v1.metadata.description,
            "model_path": self.model_v1.metadata.model_path
        }
        response = self.client.post("/register_model", json=register_payload_v1)
        self.assertEqual(response.status_code, 201)
        self.assertEqual(json.loads(response.data)["status"], "Model registered and promoted to STAGED")

        # 2. Deploy modelo v1
        deploy_payload_v1 = {
            "model_name": self.model_v1.metadata.name,
            "model_version": self.model_v1.metadata.version,
            "strategy": "BLUE_GREEN"
        }
        response = self.client.post("/deploy_model", json=deploy_payload_v1)
        self.assertEqual(response.status_code, 201)
        self.assertIn("endpoint", json.loads(response.data))

        # 3. Predição com modelo v1
        predict_payload = {"feature_1": 0.7, "feature_2": 8}
        response = self.client.post(f"/predict/{self.model_v1.metadata.name}/{self.model_v1.metadata.version}", json=predict_payload)
        self.assertEqual(response.status_code, 200)
        prediction_data = json.loads(response.data)
        self.assertIn("prediction", prediction_data)
        self.assertEqual(prediction_data["model_version"], self.model_v1.metadata.version)

        # 4. Registrar modelo v2 (nova versão do mesmo modelo)
        register_payload_v2 = {
            "name": self.model_v2.metadata.name,
            "version": self.model_v2.metadata.version,
            "framework": self.model_v2.metadata.framework,
            "author": self.model_v2.metadata.author,
            "description": self.model_v2.metadata.description,
            "model_path": self.model_v2.metadata.model_path
        }
        response = self.client.post("/register_model", json=register_payload_v2)
        self.assertEqual(response.status_code, 201)
        self.assertEqual(json.loads(response.data)["status"], "Model registered and promoted to STAGED")

        # 5. Deploy modelo v2 com estratégia Canary (deve manter v1 ativo e adicionar v2)
        deploy_payload_v2 = {
            "model_name": self.model_v2.metadata.name,
            "model_version": self.model_v2.metadata.version,
            "strategy": "CANARY",
            "replicas": 1
        }
        response = self.client.post("/deploy_model", json=deploy_payload_v2)
        self.assertEqual(response.status_code, 201)
        self.assertIn("endpoint", json.loads(response.data))

        # Verificar que ambos os modelos estão implantados
        response = self.client.get("/deployments")
        self.assertEqual(response.status_code, 200)
        deployments = json.loads(response.data)
        self.assertEqual(len(deployments), 2)
        self.assertTrue(any(d["model_version"] == "1.0.0" for d in deployments))
        self.assertTrue(any(d["model_version"] == "2.0.0" for d in deployments))

        # 6. Undeploy modelo v1
        undeploy_payload = {
            "model_name": self.model_v1.metadata.name,
            "model_version": self.model_v1.metadata.version
        }
        response = self.client.post("/undeploy_model", json=undeploy_payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.data)["status"], "Model undeployed")

        # Verificar que apenas o modelo v2 está implantado
        response = self.client.get("/deployments")
        self.assertEqual(response.status_code, 200)
        deployments = json.loads(response.data)
        self.assertEqual(len(deployments), 1)
        self.assertEqual(deployments[0]["model_version"], "2.0.0")

    def test_persistence_across_platform_instances(self):
        """Testa se o registro e os deployments persistem entre instâncias da plataforma"""
        # Registrar e implantar um modelo na primeira instância
        self.platform.registry.register_model(self.model_v1)
        self.model_v1.promote_to_staging()
        self.platform.deploy_model(self.model_v1, DeploymentConfig(strategy=DeploymentStrategy.ROLLING))

        # Criar uma nova instância da plataforma (simulando reinício)
        new_platform = DeploymentPlatform(name="test-platform-reloaded", registry_file=self.test_registry_file)
        new_platform.load_state() # Carregar estado do disco

        # Verificar se o modelo e o deployment foram carregados
        reloaded_model = new_platform.registry.get_model(self.model_v1.metadata.name, self.model_v1.metadata.version)
        self.assertIsNotNone(reloaded_model)
        self.assertEqual(reloaded_model.status, ModelStatus.PRODUCTION) # Deploy muda o status para PRODUCTION

        deployment_key = f"{self.model_v1.metadata.name}-{self.model_v1.metadata.version}"
        self.assertIn(deployment_key, new_platform.deployments)
        self.assertEqual(new_platform.deployments[deployment_key]["status"], "running")

if __name__ == '__main__':
    unittest.main(verbosity=2)

