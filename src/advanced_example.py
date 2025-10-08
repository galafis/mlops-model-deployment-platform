
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.model_deployment import DeploymentPlatform, ModelMetadata, Model, DeploymentConfig, DeploymentStrategy
from src.model_serving_api import app as model_api_app
import requests
import json
import threading
import time

# --- 0. Preparação de Dados e Treinamento de Modelo Mais Complexo ---
def prepare_and_train_model(model_dir):
    print("\n--- 0. Preparando Dados e Treinando Modelo de Exemplo Avançado ---")
    # Gerar dados sintéticos para um problema de classificação binária
    np.random.seed(42)
    data_size = 1000
    features = pd.DataFrame({
        'feature_1': np.random.rand(data_size) * 100,
        'feature_2': np.random.rand(data_size) * 50,
        'feature_3': np.random.randint(0, 2, data_size),
        'feature_4': np.random.normal(50, 10, data_size)
    })
    # Criar uma target mais complexa
    target = ((features['feature_1'] * 0.5 + features['feature_2'] * 0.8 + features['feature_3'] * 10 + features['feature_4'] * 0.2) > 100).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Treinar um RandomForestClassifier
    advanced_model = RandomForestClassifier(n_estimators=100, random_state=42)
    advanced_model.fit(X_train, y_train)

    # Avaliar o modelo
    y_pred = advanced_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"  Modelo RandomForest treinado com sucesso.")
    print(f"  Métricas: Acurácia={accuracy:.2f}, Precisão={precision:.2f}, Recall={recall:.2f}")

    # Salvar o modelo
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "advanced_churn_predictor_v1.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(advanced_model, f)
    print(f"  Modelo avançado salvo em {model_path}")
    return model_path, accuracy, precision, recall, X_test.iloc[0].to_dict(), X_train, y_train, X_test, y_test


def run_advanced_example():
    print("\n==================================================")
    print("Demonstração da Plataforma de Deploy de Modelos MLOps - Exemplo Avançado")
    print("==================================================")

    model_dir = "./models"

    # Limpar arquivos de registro e deployment para garantir um estado limpo
    if os.path.exists("model_registry.json"):
        os.remove("model_registry.json")
    if os.path.exists("model_deployments.json"):
        os.remove("model_deployments.json")

    model_path, accuracy, precision, recall, sample_input, X_train, y_train, X_test, y_test = prepare_and_train_model(model_dir)


    # --- 1. Inicializar Plataforma de Deployment ---
    print("\n--- 1. Inicializando Plataforma de Deployment ---")
    platform = DeploymentPlatform("production-platform")
    
    # --- 2. Registrar Modelo ---
    print("\n--- 2. Registrando Modelo Avançado ---")
    metadata_v1 = ModelMetadata(
        name="advanced-churn-predictor",
        version="1.0.0",
        framework="scikit-learn",
        author="gabriel.lafis@example.com",
        description="Modelo avançado para predição de churn de clientes (RandomForest)",
        metrics={"accuracy": accuracy, "precision": precision, "recall": recall},
        tags=["classification", "churn", "advanced", "production"],
        model_path=model_path
    )
    model_v1 = Model(metadata_v1)
    platform.registry.register_model(model_v1)
    print(f"  Modelo {model_v1.metadata.name} v{model_v1.metadata.version} registrado.")

    # --- 3. Promover Modelo para Staging e Produção ---
    print("\n--- 3. Promovendo Modelo para Staging e Produção ---")
    model_v1.promote_to_staging()
    model_v1.promote_to_production()

    # --- 4. Iniciar API de Serviço de Modelos (em thread separada) ---
    print("\n--- 4. Iniciando API de Serviço de Modelos ---")
    api_thread = threading.Thread(target=lambda: model_api_app.run(port=5001, debug=False, use_reloader=False))
    api_thread.daemon = True
    api_thread.start()
    time.sleep(3) # Dar um tempo para a API iniciar
    print("  API de serviço de modelos iniciada em http://127.0.0.1:5001")

    # --- 5. Realizar Deployment (Blue/Green) ---
    print("\n--- 5. Realizando Deployment Blue/Green para v1.0.0 ---")
    config_v1 = DeploymentConfig(
        strategy=DeploymentStrategy.BLUE_GREEN,
        replicas=2,
        auto_scaling=True
    )
    endpoint_v1 = platform.deploy_model(model_v1, config_v1)
    if endpoint_v1:
        print(f"  Deployment de {model_v1.metadata.name} v{model_v1.metadata.version} concluído. Endpoint: {endpoint_v1}")
    else:
        print(f"  Falha no deployment do modelo {model_v1.metadata.name} v{model_v1.metadata.version}.")
        return

    # --- 6. Realizar Previsão via API ---
    print("\n--- 6. Realizando Previsão com o Modelo Implantado (v1.0.0) ---")
    input_data = {"features": [list(sample_input.values())]} # Usar um exemplo real do X_test
    try:
        response = requests.post(f"http://127.0.0.1:5001/predict/{model_v1.metadata.name}/{model_v1.metadata.version}", json=input_data)
        response.raise_for_status()
        prediction_result = response.json()
        print(f"  Resultado da previsão (v1.0.0): {prediction_result}")
    except requests.exceptions.ConnectionError:
        print("  ERRO: Não foi possível conectar à API Flask. Verifique se ela está rodando na porta 5001.")
    except Exception as e:
        print(f"  Erro ao consultar API para previsão: {e}")

    # --- 7. Simular uma Nova Versão do Modelo (v2.0.0) e Canary Release ---
    print("\n--- 7. Simular uma Nova Versão do Modelo (v2.0.0) e Canary Release ---")
    # Treinar uma nova versão do modelo (simulando melhorias)
    advanced_model_v2 = RandomForestClassifier(n_estimators=120, random_state=42, max_depth=10) # Modelo ligeiramente diferente
    advanced_model_v2.fit(X_train, y_train)
    y_pred_v2 = advanced_model_v2.predict(X_test)
    accuracy_v2 = accuracy_score(y_test, y_pred_v2)
    precision_v2 = precision_score(y_test, y_pred_v2)
    recall_v2 = recall_score(y_test, y_pred_v2)

    model_path_v2 = os.path.join(model_dir, "advanced_churn_predictor_v2.pkl")
    with open(model_path_v2, "wb") as f:
        pickle.dump(advanced_model_v2, f)
    print(f"  Modelo avançado v2 salvo em {model_path_v2}")
    print(f"  Métricas v2: Acurácia={accuracy_v2:.2f}, Precisão={precision_v2:.2f}, Recall={recall_v2:.2f}")

    metadata_v2 = ModelMetadata(
        name="advanced-churn-predictor",
        version="2.0.0",
        framework="scikit-learn",
        author="gabriel.lafis@example.com",
        description="Modelo avançado para predição de churn de clientes (RandomForest v2 - otimizado)",
        metrics={"accuracy": accuracy_v2, "precision": precision_v2, "recall": recall_v2},
        tags=["classification", "churn", "advanced", "production", "canary"],
        model_path=model_path_v2
    )
    model_v2 = Model(metadata_v2)
    platform.registry.register_model(model_v2)
    model_v2.promote_to_staging()

    config_v2 = DeploymentConfig(
        strategy=DeploymentStrategy.CANARY,
        replicas=1,
        auto_scaling=False,
        canary_traffic_percentage=20 # 20% do tráfego para a nova versão
    )
    # Verificar se já existe um deployment ativo para model_v2
    existing_deployment_v2 = platform.deployments.get(f"{model_v2.metadata.name}-{model_v2.metadata.version}")
    if not existing_deployment_v2:
        endpoint_v2 = platform.deploy_model(model_v2, config_v2)
        if endpoint_v2:
            print(f"  Deployment Canary de {model_v2.metadata.name} v{model_v2.metadata.version} concluído. Endpoint: {endpoint_v2}")
        else:
            print(f"  Falha no deployment Canary do modelo {model_v2.metadata.name} v{model_v2.metadata.version}.")
            return
    else:
        print(f"  Modelo {model_v2.metadata.name} v{model_v2.metadata.version} já possui um deployment ativo. Reutilizando.")
        endpoint_v2 = existing_deployment_v2["endpoint"]

    # --- 8. Simular Tráfego para Canary Release ---
    print("\n--- 8. Simular Tráfego para Canary Release (v1.0.0 vs v2.0.0) ---")
    print("  Simulando 10 requisições com 20% de tráfego para v2.0.0:")
    for i in range(10):
        target_version = "1.0.0" if np.random.rand() * 100 > config_v2.canary_traffic_percentage else "2.0.0"
        target_endpoint = f"http://127.0.0.1:5001/predict/{model_v1.metadata.name}/{target_version}"
        try:
            response = requests.post(target_endpoint, json=input_data)
            response.raise_for_status()
            prediction_result = response.json()
            print(f"    Requisição {i+1}: Modelo v{prediction_result.get('model_version')} -> {prediction_result}")
        except requests.exceptions.ConnectionError:
            print(f"    Requisição {i+1}: ERRO - Não foi possível conectar à API Flask para {target_version}.")
        except Exception as e:
            print(f"    Requisição {i+1}: Erro ao consultar API para previsão (v{target_version}): {e}")
        time.sleep(0.5)

    # --- 9. Promover Canary para Produção Completa ---
    print("\n--- 9. Promovendo Canary (v2.0.0) para Produção Completa ---")
    # A função promote_canary_to_production já chama model.promote_to_production()
    platform.promote_canary_to_production(model_v2)
    print(f"  Modelo {model_v2.metadata.name} v{model_v2.metadata.version} promovido para produção completa.")
    # Atualizar o status do modelo no objeto local para refletir a mudança no registro
    model_v2.status = platform.registry.get_model(model_v2.metadata.name, model_v2.metadata.version).status
    # Recarregar a plataforma na API de serviço para que ela reconheça as mudanças
    try:
        requests.post("http://127.0.0.1:5001/reload_platform")
        print("  API de serviço de modelos recarregada com sucesso.")
    except Exception as e:
        print(f"  Erro ao recarregar a API de serviço de modelos: {e}")

    # --- 10. Realizar Previsão com o Modelo v2.0.0 em Produção ---
    print("\n--- 10. Realizando Previsão com o Modelo Implantado (v2.0.0) ---")
    try:
        response = requests.post(f"http://127.0.0.1:5001/predict/{model_v2.metadata.name}/{model_v2.metadata.version}", json=input_data)
        response.raise_for_status()
        prediction_result = response.json()
        print(f"  Resultado da previsão (v2.0.0): {prediction_result}")
    except requests.exceptions.ConnectionError:
        print("  ERRO: Não foi possível conectar à API Flask. Verifique se ela está rodando na porta 5001.")
    except Exception as e:
        print(f"  Erro ao consultar API para previsão: {e}")

    # --- 11. Fazer Undeploy da Versão Antiga (v1.0.0) ---
    print("\n--- 11. Fazendo Undeploy da Versão Antiga (v1.0.0) ---")
    platform.undeploy_model(model_v1.metadata.name, model_v1.metadata.version)
    print(f"  Modelo {model_v1.metadata.name} v{model_v1.metadata.version} desativado.")

    print("\n==================================================")
    print("Demonstração do Exemplo Avançado Concluída.")
    print("==================================================")

if __name__ == "__main__":
    run_advanced_example()

