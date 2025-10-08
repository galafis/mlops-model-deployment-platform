
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_deployment import (
    DeploymentPlatform,
    Model,
    ModelMetadata,
    ModelStatus,
    DeploymentStrategy,
    DeploymentConfig
)

class TestModelDeployment(unittest.TestCase):
    def setUp(self):
        self.platform = DeploymentPlatform(name="test-platform")
        self.metadata = ModelMetadata(
            name="test-model",
            version="1.0.0",
            framework="test-framework",
            author="test-author",
            description="A test model"
        )
        self.model = Model(self.metadata)

    def test_register_model(self):
        self.assertTrue(self.platform.registry.register_model(self.model))
        latest_model = self.platform.registry.get_latest_version("test-model")
        self.assertIsNotNone(latest_model)
        self.assertEqual(latest_model.metadata.version, "1.0.0")

    def test_promote_model(self):
        self.assertEqual(self.model.status, ModelStatus.TRAINING)
        self.assertTrue(self.model.promote_to_staging())
        self.assertEqual(self.model.status, ModelStatus.STAGED)
        self.assertTrue(self.model.promote_to_production())
        self.assertEqual(self.model.status, ModelStatus.PRODUCTION)

    def test_deploy_model(self):
        self.platform.registry.register_model(self.model)
        self.model.promote_to_staging()
        self.model.promote_to_production()
        config = DeploymentConfig(strategy=DeploymentStrategy.BLUE_GREEN)
        self.assertTrue(self.platform.deploy_model(self.model, config))
        deployment_info = self.platform.get_deployment_info("test-model")
        self.assertIsNotNone(deployment_info)
        self.assertEqual(deployment_info["status"], "running")

    def test_get_production_model(self):
        self.platform.registry.register_model(self.model)
        self.model.promote_to_staging()
        self.model.promote_to_production()
        prod_model = self.platform.registry.get_production_model("test-model")
        self.assertIsNotNone(prod_model)
        self.assertEqual(prod_model.metadata.version, "1.0.0")

if __name__ == '__main__':
    unittest.main(verbosity=2)

