import streamlit as st
import pandas as pd

prod_file = '../data/processed/prediction_prod.parquet'
dev_file = '../data/processed/prediction_test.parquet'
############################################ SIDE BAR TITLE
st.sidebar.title('Painel de Controle')
st.sidebar.markdown(f"""
Previsões do modelo de arremessos a partir  das métricas de desempenho do modelo em diferentes conjuntos de dados ( dados de desenvolvimento e produção do modelo).
""")

df_prod = pd.read_parquet('../data/processed/prediction_prod.parquet')
df_dev = pd.read_parquet('../data/processed/prediction_test.parquet')

# Aplicar um threshold
threshold = 0.5
df_prod['prediction_label'] = (df_prod['predict_score'] >= threshold).astype(int)

from sklearn import metrics

# Calcular as métricas de classificação
metrics_dev = metrics.precision_recall_fscore_support(df_dev['shot_made_flag'], df_dev['prediction_label'], average='binary')
metrics_prod = metrics.precision_recall_fscore_support(df_prod['shot_made_flag'], df_prod['prediction_label'], average='binary')
accuracy_dev = metrics.accuracy_score(df_dev['shot_made_flag'], df_dev['prediction_label'])
accuracy_prod = metrics.accuracy_score(df_prod['shot_made_flag'], df_prod['prediction_label'])

# Organizar as métricas em um DataFrame
df_metrics = pd.DataFrame({
    'Métrica': ['Precisão', 'Revocação', 'F1-Score', 'Acurácia'],
    'Desenvolvimento': [metrics_dev[0], metrics_dev[1], metrics_dev[2], accuracy_dev],
    'Produção': [metrics_prod[0], metrics_prod[1], metrics_prod[2], accuracy_prod]
})

# Exibir a tabela de métricas
st.write(df_metrics)


