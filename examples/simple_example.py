"""
Simple example of using the MLOps Model Deployment Platform

This example demonstrates the basic workflow of:
1. Creating a simple model
2. Registering it in the platform
3. Deploying it
4. Making predictions
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model_deployment import (
    DeploymentPlatform,
    Model,
    ModelMetadata,
    DeploymentConfig,
    DeploymentStrategy,
)


def main():
    print("=" * 60)
    print("MLOps Platform - Simple Example")
    print("=" * 60)

    # 1. Initialize the deployment platform
    print("\n1. Initializing deployment platform...")
    platform = DeploymentPlatform("simple-example-platform", registry_file="example_registry.json")

    # 2. Create model metadata
    print("\n2. Creating model metadata...")
    metadata = ModelMetadata(
        name="simple-classifier",
        version="1.0.0",
        framework="scikit-learn",
        author="example@mlops.com",
        description="Simple classification model for demonstration",
        metrics={"accuracy": 0.92, "precision": 0.89, "recall": 0.91},
        tags=["classification", "demo", "simple"],
    )

    # 3. Create and register the model
    print("\n3. Creating and registering model...")
    model = Model(metadata)
    platform.registry.register_model(model)

    # 4. Promote to staging
    print("\n4. Promoting model to staging...")
    model.promote_to_staging()
    platform.save_registry()

    # 5. Promote to production
    print("\n5. Promoting model to production...")
    model.promote_to_production()
    platform.save_registry()

    # 6. Create deployment configuration
    print("\n6. Creating deployment configuration...")
    config = DeploymentConfig(
        strategy=DeploymentStrategy.BLUE_GREEN, replicas=2, auto_scaling=True, min_replicas=1, max_replicas=5
    )

    # 7. Deploy the model
    print("\n7. Deploying model...")
    endpoint = platform.deploy_model(model, config)
    print(f"   Model deployed at: {endpoint}")

    # 8. Make a prediction
    print("\n8. Making a prediction...")
    input_data = {"features": [[0.5, 0.3, 0.8, 0.2]]}
    prediction = platform.predict("simple-classifier", "1.0.0", input_data)
    print(f"   Prediction result: {prediction}")

    # 9. List all models
    print("\n9. Listing all registered models...")
    models = platform.registry.list_models()
    for m in models:
        print(f"   - {m['name']} v{m['version']} ({m['status']}) - {m['framework']}")

    # 10. Scale deployment
    print("\n10. Scaling deployment to 3 replicas...")
    platform.scale_deployment("simple-classifier", "1.0.0", 3)

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)

    # Cleanup
    print("\nCleaning up example files...")
    if os.path.exists("example_registry.json"):
        os.remove("example_registry.json")
    if os.path.exists("model_deployments.json"):
        os.remove("model_deployments.json")
    print("Done!")


if __name__ == "__main__":
    main()
