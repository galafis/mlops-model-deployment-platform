"""
MLOps Model Deployment Platform
Author: Gabriel Demetrios Lafis
Year: 2025

Sistema para gerenciamento e deployment de modelos de ML em produção.
"""

from .model_deployment import (
    DeploymentPlatform,
    Model,
    ModelMetadata,
    ModelRegistry,
    ModelStatus,
    DeploymentStrategy,
    DeploymentConfig
)

__all__ = [
    'DeploymentPlatform',
    'Model',
    'ModelMetadata',
    'ModelRegistry',
    'ModelStatus',
    'DeploymentStrategy',
    'DeploymentConfig'
]

__version__ = '1.0.0'
