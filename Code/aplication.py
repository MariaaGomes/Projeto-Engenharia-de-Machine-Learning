
import pandas as pd
import numpy as np
import pycaret.classification as pc

import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Para usar o sqlite como repositorio
mlflow.set_tracking_uri("sqlite:///mlruns.db")

experiment_name = 'Treinamento Kobe'
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    experiment = mlflow.get_experiment(experiment_id)
experiment_id = experiment.experiment_id


data_cols = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs','shot_distance']

with mlflow.start_run(experiment_id=experiment_id, run_name = 'PipelineAplicacao'):

    model_uri = f"models:/model_kobe@staging"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    data_prod = pd.read_parquet('../data/raw/dataset_kobe_prod.parquet')
    
    # Remover linhas com valores nulos em qualquer coluna
    data_prod = data_prod.dropna()
    
    Y = loaded_model.predict_proba(data_prod[data_cols])[:,1]
    data_prod['predict_score'] = Y

    data_prod.to_parquet('../data/processed/prediction_prod.parquet')
    mlflow.log_artifact('../data/processed/prediction_prod.parquet')
    
    # Calcular a matriz de confusão
    y_true = data_prod['shot_made_flag']  
    y_pred = data_prod['predict_score'] 

    # Definir um limiar de decisão (por exemplo, 0.5)
    threshold = 0.5

    # Converter as probabilidades em valores binários
    y_pred_binary = (y_pred > threshold).astype(int)

    # Calcular a matriz de confusão
    cm = confusion_matrix(y_true, y_pred_binary)

    # Plotar a matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.gcf()
    plt.show()
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')

    # Calcular métricas de avaliação
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)

    mlflow.log_metric('f1_score', f1)

    # Imprimir métricas de avaliação
    print(f'Acurácia: {accuracy:.2f}')
    print(f'Precisão: {precision:.2f}')
    print(f'Revocação: {recall:.2f}')
    print(f'F1-Score: {f1:.2f}')

    print(data_prod)
    


