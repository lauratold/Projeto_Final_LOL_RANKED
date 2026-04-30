import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    classification_report, 
    roc_auc_score, 
    roc_curve,
    accuracy_score
)

# Carregamento de Dados

df = pd.read_csv('C:\\Users\\Laura\\OneDrive\\Desktop\\Base_LOL_RANKED_WIN.csv')

print(df.head())


# Verificação de Nulos
print("Soma de valores nulos:\n", df.isnull().sum().sum())
''
print(df.info())
# Deletar colunas irrelevantes (ID do jogo não tem valor preditivo)
if 'gameId' in df.columns:
    df.drop('gameId', axis=1, inplace=True)

# Tratamento de Outliers

print(df.describe())

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(data=df[['blueWardsPlaced', 'blueKills', 'blueTotalGold']])
plt.title("Outliers Originais")

def remover_outliers(data, colunas):
    df_out = data.copy()
    for col in colunas:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        df_out = df_out[(df_out[col] >= Q1 - 1.5*IQR) & (df_out[col] <= Q3 + 1.5*IQR)]
    return df_out

# Aplicando limpeza em colunas críticas
colunas_criticas = ['blueWardsPlaced', 'blueKills', 'blueTotalGold', 'blueGoldDiff']
df = remover_outliers(df, colunas_criticas)

plt.subplot(1, 2, 2)
sns.boxplot(data=df[['blueWardsPlaced', 'blueKills', 'blueTotalGold']])
plt.title("Pós-Tratamento (IQR)")
plt.show()

# Binarização
# Criei uma base nova baseada em conceitos binários para melhor avaliação estratégica

df_final = pd.DataFrame()
df_final['blueWins'] = df['blueWins']

# Transformação para Binário: 1 se vantagem/evento ocorreu, 0 se não
df_final['blueGoldLead'] = (df['blueGoldDiff'] > 0).astype(int)
df_final['blueExpLead'] = (df['blueExperienceDiff'] > 0).astype(int)
df_final['blueFirstBlood'] = df['blueFirstBlood'] # Já é binário
df_final['blueHasDragon'] = (df['blueDragons'] > 0).astype(int)
df_final['blueHasHerald'] = (df['blueHeralds'] > 0).astype(int)
df_final['blueTowerDestroyed'] = (df['blueTowersDestroyed'] > 0).astype(int)
df_final['blueHighKills'] = (df['blueKills'] > df['blueKills'].median()).astype(int)

# Análise de Correlação para variáveis binárias
plt.figure(figsize=(10, 8))
corr = df_final.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlação (Variáveis Binárias)")
plt.show()

# Filtragem: Se a correlação com o alvo (blueWins) for muito baixa (< 0.1), excluímos
target_corr = corr['blueWins'].abs()
relevantes = target_corr[target_corr > 0.1].index.tolist()
df_final = df_final[relevantes]

print(f"Variáveis selecionadas por relevância: {relevantes}")

# Preparação, Padronização e Divisão
X = df_final.drop('blueWins', axis=1)
y = df_final['blueWins']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelagem
# Usando Grid Search para encontrar a melhor árvore
params = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_leaf': [5, 10, 20],
    'criterion': ['gini', 'entropy']
}

grid = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=5, scoring='roc_auc')
grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_

# Avaliação final e Métricas
print("\n--- Gerando Métricas Finais ---")

# Previsões usando o melhor modelo do GridSearch
y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

# (Precision, Recall, F1)
print("\n--- Relatório de Performance ---")
print(classification_report(y_test, y_pred, target_names=['Derrota', 'Vitória']))

# 2. Matriz de Confusão Visual
fig, ax = plt.subplots(figsize=(6, 6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Derrota', 'Vitória'])
disp.plot(cmap='Blues', ax=ax)
plt.title("Matriz de Confusão: Predição de Vitória (Blue)")
plt.show()

# 3. Curva ROC e AUC
auc_score = roc_auc_score(y_test, y_proba)
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_score:.4f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - Desempenho da Árvore de Decisão')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

print(f"\nProjeto Finalizado com Sucesso! AUC Final: {auc_score:.4f}")
