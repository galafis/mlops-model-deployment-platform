
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


class TestModelDeployment(unittest.TestCase):
    def setUp(self):
        self.test_registry_file = "test_model_registry.json"
        self.test_deployments_file = "test_model_deployments.json"

        # Limpar arquivos de teste antes de cada execução
        if os.path.exists(self.test_registry_file):
            os.remove(self.test_registry_file)
        if os.path.exists(self.test_deployments_file):
            os.remove(self.test_deployments_file)

        self.platform = DeploymentPlatform(name="test-platform", registry_file=self.test_registry_file)

        self.metadata_v1 = ModelMetadata(
            name="test-model",
            version="1.0.0",
            framework="scikit-learn",
            author="test-author",
            description="A test model v1",
            model_path="./models/test_v1.pkl"
        )
        self.model_v1 = Model(self.metadata_v1)

        self.metadata_v2 = ModelMetadata(
            name="test-model",
            version="2.0.0",
            framework="tensorflow",
            author="test-author",
            description="A test model v2",
            model_path="./models/test_v2.h5"
        )
        self.model_v2 = Model(self.metadata_v2)

        self.metadata_other = ModelMetadata(
            name="other-model",
            version="1.0.0",
            framework="pytorch",
            author="other-author",
            description="Another test model",
            model_path="./models/other_v1.pt"
        )
        self.model_other = Model(self.metadata_other)

    def tearDown(self):
        # Limpar arquivos de teste após cada execução
        if os.path.exists(self.test_registry_file):
            os.remove(self.test_registry_file)
        if os.path.exists(self.test_deployments_file):
            os.remove(self.test_deployments_file)

    def test_register_model(self):
        self.assertTrue(self.platform.registry.register_model(self.model_v1))
        self.assertIsNotNone(self.platform.registry.get_model("test-model", "1.0.0"))
        self.assertEqual(len(self.platform.registry.models["test-model"]), 1)

        self.assertTrue(self.platform.registry.register_model(self.model_v2))
        self.assertIsNotNone(self.platform.registry.get_model("test-model", "2.0.0"))
        self.assertEqual(len(self.platform.registry.models["test-model"]), 2)

        # Tentar registrar versão duplicada
        self.assertFalse(self.platform.registry.register_model(self.model_v1))

    def test_model_status_transitions(self):
        self.assertEqual(self.model_v1.status, ModelStatus.TRAINING)
        self.assertTrue(self.model_v1.promote_to_staging())
        self.assertEqual(self.model_v1.status, ModelStatus.STAGED)
        self.assertTrue(self.model_v1.promote_to_production())
        self.assertEqual(self.model_v1.status, ModelStatus.PRODUCTION)
        self.assertTrue(self.model_v1.archive_model())
        self.assertEqual(self.model_v1.status, ModelStatus.ARCHIVED)

        # Não deve ser possível promover de ARCHIVED para STAGED diretamente
        self.assertFalse(self.model_v1.promote_to_staging())

    def test_deploy_model_blue_green(self):
        self.platform.registry.register_model(self.model_v1)
        self.model_v1.promote_to_staging()
        self.model_v1.promote_to_production() # Status muda para PRODUCTION após deploy
        config = DeploymentConfig(strategy=DeploymentStrategy.BLUE_GREEN)
        endpoint = self.platform.deploy_model(self.model_v1, config)
        self.assertIsNotNone(endpoint)
        self.assertIn(f"test-model-1.0.0", self.platform.deployments)
        self.assertEqual(self.platform.deployments[f"test-model-1.0.0"]["status"], "running")
        self.assertEqual(self.model_v1.status, ModelStatus.PRODUCTION)

    def test_deploy_model_canary(self):
        self.platform.registry.register_model(self.model_v2)
        self.model_v2.promote_to_staging()
        config = DeploymentConfig(strategy=DeploymentStrategy.CANARY, replicas=1)
        endpoint = self.platform.deploy_model(self.model_v2, config)
        self.assertIsNotNone(endpoint)
        self.assertIn(f"test-model-2.0.0", self.platform.deployments)
        self.assertEqual(self.platform.deployments[f"test-model-2.0.0"]["status"], "running")
        self.assertEqual(self.model_v2.status, ModelStatus.PRODUCTION)

    def test_undeploy_model(self):
        self.platform.registry.register_model(self.model_v1)
        self.model_v1.promote_to_staging()
        self.model_v1.promote_to_production()
        config = DeploymentConfig(strategy=DeploymentStrategy.ROLLING)
        self.platform.deploy_model(self.model_v1, config)

        self.assertTrue(self.platform.undeploy_model("test-model", "1.0.0"))
        self.assertNotIn(f"test-model-1.0.0", self.platform.deployments)
        self.assertEqual(self.model_v1.status, ModelStatus.ARCHIVED)

    def test_scale_deployment(self):
        self.platform.registry.register_model(self.model_v1)
        self.model_v1.promote_to_staging()
        self.model_v1.promote_to_production()
        config = DeploymentConfig(strategy=DeploymentStrategy.ROLLING, replicas=1)
        self.platform.deploy_model(self.model_v1, config)

        self.assertTrue(self.platform.scale_deployment("test-model", "1.0.0", 5))
        self.assertEqual(self.platform.deployments[f"test-model-1.0.0"]["config"].replicas, 5)

    def test_predict(self):
        self.platform.registry.register_model(self.model_v1)
        self.model_v1.promote_to_staging()
        self.model_v1.promote_to_production()
        config = DeploymentConfig(strategy=DeploymentStrategy.BLUE_GREEN)
        self.platform.deploy_model(self.model_v1, config)

        input_data = {"feature_1": 0.8, "feature_2": 10}
        prediction = self.platform.predict("test-model", "1.0.0", input_data)
        self.assertIsNotNone(prediction)
        self.assertIn("prediction", prediction)
        self.assertEqual(prediction["model_version"], "1.0.0")

    def test_model_registry_persistence(self):
        # Registrar modelos e salvar
        self.platform.registry.register_model(self.model_v1)
        self.model_v1.promote_to_staging()
        self.model_v1.promote_to_production()
        self.platform.registry.register_model(self.model_other)
        self.model_other.promote_to_staging()

        # Criar nova plataforma para carregar do arquivo
        new_platform = DeploymentPlatform(name="new-platform", registry_file=self.test_registry_file)
        
        # Verificar se os modelos foram carregados
        self.assertIsNotNone(new_platform.registry.get_model("test-model", "1.0.0"))
        self.assertEqual(new_platform.registry.get_model("test-model", "1.0.0").status, ModelStatus.PRODUCTION)
        self.assertIsNotNone(new_platform.registry.get_model("other-model", "1.0.0"))
        self.assertEqual(new_platform.registry.get_model("other-model", "1.0.0").status, ModelStatus.STAGED)

    def test_deployment_persistence(self):
        # Registrar e implantar modelos
        self.platform.registry.register_model(self.model_v1)
        self.model_v1.promote_to_staging()
        self.platform.deploy_model(self.model_v1, DeploymentConfig(strategy=DeploymentStrategy.BLUE_GREEN))

        self.platform.registry.register_model(self.model_other)
        self.model_other.promote_to_staging()
        self.platform.deploy_model(self.model_other, DeploymentConfig(strategy=DeploymentStrategy.CANARY))

        # Criar nova plataforma para carregar deployments
        new_platform = DeploymentPlatform(name="new-platform", registry_file=self.test_registry_file)
        new_platform.deployments = new_platform._load_deployments() # Forçar recarregamento

        # Verificar se os deployments foram carregados
        self.assertIn(f"test-model-1.0.0", new_platform.deployments)
        self.assertIn(f"other-model-1.0.0", new_platform.deployments)
        self.assertEqual(new_platform.deployments[f"test-model-1.0.0"]["status"], "running")

    def test_flask_api_predict_endpoint(self):
        # Mock Flask app
        if not hasattr(self.platform, 'create_flask_api'):
            self.skipTest("Flask not installed or create_flask_api not available")

        app = self.platform.create_flask_api()
        app.testing = True
        client = MockFlaskClient(app)

        self.platform.registry.register_model(self.model_v1)
        self.model_v1.promote_to_staging()
        self.platform.deploy_model(self.model_v1, DeploymentConfig(strategy=DeploymentStrategy.BLUE_GREEN))

        input_data = {"feature_1": 0.6, "feature_2": 5}
        response = client.post("/predict/test-model/1.0.0", json=input_data)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["prediction"], 1)
        self.assertEqual(data["model_version"], "1.0.0")

    def test_flask_api_list_models_endpoint(self):
        if not hasattr(self.platform, 'create_flask_api'):
            self.skipTest("Flask not installed or create_flask_api not available")

        app = self.platform.create_flask_api()
        app.testing = True
        client = MockFlaskClient(app)

        self.platform.registry.register_model(self.model_v1)
        self.platform.registry.register_model(self.model_v2)

        response = client.get("/models")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["name"], "test-model")
        self.assertEqual(data[1]["version"], "2.0.0")

    def test_flask_api_list_deployments_endpoint(self):
        if not hasattr(self.platform, 'create_flask_api'):
            self.skipTest("Flask not installed or create_flask_api not available")

        app = self.platform.create_flask_api()
        app.testing = True
        client = MockFlaskClient(app)

        self.platform.registry.register_model(self.model_v1)
        self.model_v1.promote_to_staging()
        self.platform.deploy_model(self.model_v1, DeploymentConfig(strategy=DeploymentStrategy.BLUE_GREEN))

        response = client.get("/deployments")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["model_name"], "test-model")
        self.assertEqual(data[0]["model_version"], "1.0.0")


if __name__ == '__main__':
    unittest.main(verbosity=2)

