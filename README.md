# MLOps Model Deployment Platform

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3.x-black?style=for-the-badge&logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Container-Docker-blue?style=for-the-badge&logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Orchestration-Kubernetes-blue?style=for-the-badge&logo=kubernetes&logoColor=white)
![Mermaid](https://img.shields.io/badge/Diagrams-Mermaid-orange?style=for-the-badge&logo=mermaid&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

---

## 🇧🇷 Plataforma de Deploy de Modelos MLOps

Este repositório apresenta uma **plataforma completa para o deploy e gerenciamento de modelos de Machine Learning (ML) em produção**, seguindo as melhores práticas de MLOps. O foco é em automatizar o ciclo de vida do modelo, desde o treinamento e versionamento até a implantação, monitoramento e retreinamento, garantindo **escalabilidade, confiabilidade e reprodutibilidade** em ambientes de produção.

### 🎯 Objetivo

O principal objetivo deste projeto é **fornecer um guia detalhado e exemplos de código funcional** para engenheiros de ML, cientistas de dados e arquitetos que buscam construir ou otimizar suas pipelines de MLOps. Serão abordados os conceitos fundamentais, ferramentas e tecnologias para criar uma plataforma robusta de deploy de modelos, com ênfase em **versionamento de modelos, estratégias de deployment avançadas e uma API de inferência em tempo real**.

### ✨ Destaques

- **Versionamento de Modelos**: Implementação de um registro de modelos (`ModelRegistry`) que suporta versionamento (`ModelMetadata`), permitindo o gerenciamento de diferentes versões de modelos e seus metadados associados.
- **Estratégias de Deployment Avançadas**: Suporte a diversas estratégias de deployment, como **Blue/Green** e **Canary Releases**, para garantir transições seguras e controladas de modelos em produção, minimizando riscos e tempo de inatividade.
- **API de Inferência em Tempo Real (Flask)**: Uma API RESTful construída com **Flask** para servir previsões de modelos implantados, permitindo que aplicações consumam os modelos com baixa latência e alta disponibilidade.
- **Monitoramento e Escalabilidade**: Mecanismos para simular o monitoramento de modelos em produção e a capacidade de escalar deployments (`scale_deployment`) para lidar com cargas de trabalho variáveis, garantindo resiliência e performance.
- **Automação Completa**: Demonstração de como automatizar o ciclo de vida do modelo, desde o registro até o deploy e undeploy, seguindo princípios de CI/CD.
- **Código Profissional**: Exemplos de código bem estruturados, seguindo as melhores práticas da indústria, com foco em modularidade, reusabilidade e manutenibilidade.
- **Documentação Completa**: Cada componente da plataforma é acompanhado de documentação detalhada, diagramas explicativos e casos de uso práticos.
- **Testes Incluídos**: Módulos de código validados através de testes unitários e de integração, garantindo a robustez e a confiabilidade das soluções.

### 🚀 Benefícios do MLOps em Ação

A implementação de práticas de MLOps traz uma série de benefícios cruciais para o desenvolvimento e operação de modelos de ML em escala. Este projeto ilustra como esses benefícios são alcançados:

1.  **Ciclo de Vida Acelerado:** A automação do registro, deployment e monitoramento de modelos acelera o tempo de lançamento de novos modelos e atualizações.

2.  **Confiabilidade e Estabilidade:** Estratégias de deployment como Blue/Green e Canary garantem que novas versões de modelos sejam introduzidas com segurança, minimizando o impacto em caso de falhas.

3.  **Reprodutibilidade:** O versionamento de modelos e a gestão de metadados permitem a reprodução exata de deployments anteriores, essencial para auditorias e depuração.

4.  **Colaboração Aprimorada:** A plataforma fornece uma interface padronizada para cientistas de dados e engenheiros de ML interagirem com o ciclo de vida do modelo.

5.  **Monitoramento Contínuo:** Embora simulado, o framework prevê a integração de ferramentas de monitoramento para detectar problemas de performance e *drift* de dados/modelo, acionando ações corretivas.

6.  **Governança e Conformidade:** O registro de modelos e o rastreamento de versões fornecem a base para uma governança robusta e conformidade com regulamentações.

---

## 🇬🇧 MLOps Model Deployment Platform

This repository presents a **complete platform for deploying and managing Machine Learning (ML) models in production**, following MLOps best practices. The focus is on automating the model lifecycle, from training and versioning to deployment, monitoring, and retraining, ensuring **scalability, reliability, and reproducibility** in production environments.

### 🎯 Objective

The main objective of this project is to **provide a detailed guide and functional code examples** for ML engineers, data scientists, and architects looking to build or optimize their MLOps pipelines. It will cover fundamental concepts, tools, and technologies to create a robust model deployment platform, with an emphasis on **model versioning, advanced deployment strategies, and a real-time inference API**.

### ✨ Highlights

- **Model Versioning**: Implementation of a `ModelRegistry` that supports versioning (`ModelMetadata`), allowing the management of different model versions and their associated metadata.
- **Advanced Deployment Strategies**: Support for various deployment strategies, such as **Blue/Green** and **Canary Releases**, to ensure safe and controlled transitions of models in production, minimizing risks and downtime.
- **Real-time Inference API (Flask)**: A RESTful API built with **Flask** to serve predictions from deployed models, allowing applications to consume models with low latency and high availability.
- **Monitoring and Scalability**: Mechanisms to simulate monitoring of models in production and the ability to scale deployments (`scale_deployment`) to handle varying workloads, ensuring resilience and performance.
- **Full Automation**: Demonstration of how to automate the model lifecycle, from registration to deployment and undeployment, following CI/CD principles.
- **Professional Code**: Well-structured code examples, following industry best practices, with a focus on modularity, reusability, and maintainability.
- **Complete Documentation**: Each platform component is accompanied by detailed documentation, explanatory diagrams, and practical use cases.
- **Tests Included**: Code modules validated through unit and integration tests, ensuring the robustness and reliability of the solutions.

### 📊 Visualization

![MLOps Deployment Architecture](diagrams/mlops_deployment_architecture.png)

*Diagrama ilustrativo da arquitetura da Plataforma de Deploy de Modelos MLOps, destacando os principais componentes e o fluxo de trabalho.*


---

## 🛠️ Tecnologias Utilizadas / Technologies Used

| Categoria         | Tecnologia      | Descrição                                                                 |
| :---------------- | :-------------- | :------------------------------------------------------------------------ |
| **Linguagem**     | Python          | Linguagem principal para desenvolvimento da plataforma MLOps e API.       |
| **Framework Web** | Flask           | Utilizado para construir a API RESTful de inferência de modelos.          |
| **Contêineres**   | Docker          | Para empacotar modelos e suas dependências, garantindo portabilidade.     |
| **Orquestração**  | Kubernetes      | (Conceitual) Para orquestração e gerenciamento de deployments em escala.  |
| **Versionamento** | MLflow          | (Conceitual) Para rastreamento de experimentos e registro de modelos.     |
| **Serialização**  | Pickle / JSON   | Para persistência de modelos e comunicação da API.                        |
| **Testes**        | `unittest`      | Framework de testes padrão do Python para validação de funcionalidades.   |
| **Diagramação**   | Mermaid         | Para criação de diagramas de arquitetura e fluxo de trabalho no README.   |

---

## 📁 Repository Structure

```
mlops-model-deployment-platform/
├── src/
│   ├── __init__.py
│   ├── model_deployment.py      # Lógica principal da plataforma de deployment
│   └── model_serving_api.py     # Implementação da API Flask para inferência
├── data/                        # Dados de exemplo e modelos pré-treinados
├── images/                      # Imagens e diagramas para o README e documentação
├── tests/                       # Testes unitários e de integração
├── docs/                        # Documentação adicional, guias e whitepapers sobre MLOps
├── config/                      # Arquivos de configuração (ex: para ambiente de deploy)
├── models/                      # Diretório para armazenar modelos versionados
├── requirements.txt             # Dependências Python
└── README.md                    # Este arquivo
```

---

## 🚀 Getting Started

Para começar, clone o repositório e explore os diretórios `src/` e `docs/` para exemplos detalhados e instruções de uso. Certifique-se de ter as dependências necessárias instaladas.

### Pré-requisitos

- Python 3.9+
- `pip` (gerenciador de pacotes Python)
- `scikit-learn` (para o modelo de exemplo)

### Instalação

```bash
git clone https://github.com/GabrielDemetriosLafis/mlops-model-deployment-platform.git
cd mlops-model-deployment-platform

# Instalar dependências Python
pip install -r requirements.txt
```

### Exemplo de Uso Avançado (Python)

O exemplo abaixo demonstra a inicialização da `DeploymentPlatform`, o registro de um modelo, a promoção entre ambientes (staging/production), o deployment com estratégias avançadas e a interação com a API de inferência. Este código ilustra o ciclo de vida completo de um modelo em um ambiente MLOps.

```python
from src.model_deployment import DeploymentPlatform, ModelMetadata, Model, DeploymentConfig, DeploymentStrategy
from src.model_serving_api import app as model_api_app # Importa o aplicativo Flask
import requests
import json
import threading
import time

# Exemplo de uso
if __name__ == "__main__":
    print("\n==================================================")
    print("Demonstração da Plataforma de Deploy de Modelos MLOps")
    print("==================================================")

    # --- 0. Treinar e Salvar um Modelo de Exemplo ---
    # Em um cenário real, isso viria de um pipeline de treinamento.
    # Para este exemplo, vamos criar um modelo dummy.
    print("\n--- 0. Treinando e Salvando Modelo de Exemplo ---")
    try:
        from sklearn.linear_model import LogisticRegression
        import pickle
        import numpy as np

        # Criar um modelo dummy
        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y_train = np.array([0, 0, 1, 1, 1])
        dummy_model = LogisticRegression()
        dummy_model.fit(X_train, y_train)

        # Salvar o modelo
        model_path = "./models/dummy_model_v1.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(dummy_model, f)
        print(f"  Modelo dummy salvo em {model_path}")
    except ImportError:
        print("  Scikit-learn não instalado. Pulando a criação do modelo dummy.")
        print("  Por favor, instale com: pip install scikit-learn")
        exit() # Sair se scikit-learn não estiver disponível

    # --- 1. Inicializar Plataforma de Deployment ---
    print("\n--- 1. Inicializando Plataforma de Deployment ---")
    platform = DeploymentPlatform("production-platform")
    
    # --- 2. Registrar Modelo ---
    print("\n--- 2. Registrando Modelo ---")
    metadata_v1 = ModelMetadata(
        name="customer-churn-predictor",
        version="1.0.0",
        framework="scikit-learn",
        author="data-science-team@company.com",
        description="Modelo para predição de churn de clientes (v1)",
        metrics={"accuracy": 0.92, "precision": 0.89, "recall": 0.91},
        tags=["classification", "churn", "production"],
        model_path="./models/dummy_model_v1.pkl"
    )
    model_v1 = Model(metadata_v1)
    platform.registry.register_model(model_v1)
    print(f"  Modelo {model_v1.metadata.name} v{model_v1.metadata.version} registrado.")

    # --- 3. Promover Modelo para Staging e Produção ---
    print("\n--- 3. Promovendo Modelo para Staging e Produção ---")
    model_v1.promote_to_staging()
    print(f"  Modelo v1 status: {model_v1.status.value}")
    model_v1.promote_to_production()
    print(f"  Modelo v1 status: {model_v1.status.value}")

    # --- 4. Iniciar API de Serviço de Modelos (em thread separada) ---
    print("\n--- 4. Iniciando API de Serviço de Modelos ---")
    # A API precisa ser executada em uma thread separada para não bloquear o script principal
    api_thread = threading.Thread(target=lambda: model_api_app.run(port=5000, debug=False, use_reloader=False))
    api_thread.daemon = True # Permite que a thread seja encerrada quando o programa principal sair
    api_thread.start()
    time.sleep(2) # Dar um tempo para a API iniciar
    print("  API de serviço de modelos iniciada em http://127.0.0.1:5000")

    # --- 5. Realizar Deployment (Blue/Green) ---
    print("\n--- 5. Realizando Deployment Blue/Green para v1.0.0 ---")
    config_v1 = DeploymentConfig(
        strategy=DeploymentStrategy.BLUE_GREEN,
        replicas=3,
        auto_scaling=True
    )
    platform.deploy_model(model_v1, config_v1)
    print(f"  Deployment de {model_v1.metadata.name} v{model_v1.metadata.version} iniciado com estratégia {config_v1.strategy.value}.")
    time.sleep(1) # Simular tempo de deployment
    info_v1 = platform.get_deployment_info(model_v1.metadata.name, model_v1.metadata.version)
    if info_v1:
        print(f"  Status do Deployment v1: {info_v1.get("status")}")

    # --- 6. Realizar Previsão via API ---
    print("\n--- 6. Realizando Previsão com o Modelo Implantado (v1.0.0) ---")
    input_data = {"features": [[0.7, 10]]} # Exemplo de entrada para o modelo dummy
    try:
        response = requests.post("http://127.0.0.1:5000/predict/customer-churn-predictor/1.0.0", json=input_data)
        response.raise_for_status()
        prediction_result = response.json()
        print(f"  Resultado da previsão (v1.0.0): {prediction_result}")
    except requests.exceptions.ConnectionError:
        print("  ERRO: Não foi possível conectar à API Flask. Verifique se ela está rodando.")
    except Exception as e:
        print(f"  Erro ao consultar API para previsão: {e}")

    # --- 7. Registrar e Implantar uma Nova Versão (v2.0.0) com Canary Release ---
    print("\n--- 7. Registrando e Implantando Nova Versão (v2.0.0) com Canary Release ---")
    # Criar um modelo dummy v2
    model_path_v2 = "./models/dummy_model_v2.pkl"
    dummy_model_v2 = LogisticRegression()
    dummy_model_v2.fit(X_train, y_train) # Usando os mesmos dados para simplicidade
    with open(model_path_v2, "wb") as f:
        pickle.dump(dummy_model_v2, f)
    print(f"  Modelo dummy v2 salvo em {model_path_v2}")

    metadata_v2 = ModelMetadata(
        name="customer-churn-predictor",
        version="2.0.0",
        framework="scikit-learn",
        author="data-science-team@company.com",
        description="Modelo para predição de churn de clientes (v2 - melhorado)",
        metrics={"accuracy": 0.95, "precision": 0.93, "recall": 0.94},
        tags=["classification", "churn", "production"],
        model_path="./models/dummy_model_v2.pkl"
    )
    model_v2 = Model(metadata_v2)
    platform.registry.register_model(model_v2)
    model_v2.promote_to_production()

    config_v2 = DeploymentConfig(
        strategy=DeploymentStrategy.CANARY,
        replicas=1, # Iniciar com 1 réplica para canary
        auto_scaling=False,
        canary_traffic_percentage=10 # 10% do tráfego para a nova versão
    )
    platform.deploy_model(model_v2, config_v2)
    print(f"  Deployment de {model_v2.metadata.name} v{model_v2.metadata.version} iniciado com estratégia {config_v2.strategy.value}.")
    time.sleep(1) # Simular tempo de deployment
    info_v2 = platform.get_deployment_info(model_v2.metadata.name, model_v2.metadata.version)
    if info_v2:
        print(f"  Status do Deployment v2 (Canary): {info_v2.get("status")}")

    # --- 8. Escalar Deployment (para v1.0.0) ---
    print("\n--- 8. Escalando Deployment de v1.0.0 para 5 réplicas ---")
    platform.scale_deployment(model_v1.metadata.name, model_v1.metadata.version, 5)
    print(f"  Deployment de {model_v1.metadata.name} v{model_v1.metadata.version} escalado para 5 réplicas (simulado).")

    # --- 9. Desimplantar Modelo (v1.0.0) ---
    print("\n--- 9. Desimplantando Modelo v1.0.0 ---")
    platform.undeploy_model(model_v1.metadata.name, model_v1.metadata.version)
    info_v1_after_undeploy = platform.get_deployment_info(model_v1.metadata.name, model_v1.metadata.version)
    if not info_v1_after_undeploy:
        print(f"  Modelo {model_v1.metadata.name} v{model_v1.metadata.version} desimplantado com sucesso.")

    # --- 10. Desimplantar Modelo (v2.0.0) ---
    print("\n--- 10. Desimplantando Modelo v2.0.0 ---")
    platform.undeploy_model(model_v2.metadata.name, model_v2.metadata.version)
    info_v2_after_undeploy = platform.get_deployment_info(model_v2.metadata.name, model_v2.metadata.version)
    if not info_v2_after_undeploy:
        print(f"  Modelo {model_v2.metadata.name} v{model_v2.metadata.version} desimplantado com sucesso.")

    print("\n==================================================")
    print("Demonstração Concluída.")
    print("==================================================")
```

---

## 🤝 Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues, enviar pull requests ou sugerir melhorias. Por favor, siga as diretrizes de contribuição.

---

## 📝 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

**Autor:** Gabriel Demetrios Lafis  \n**Ano:** 2025