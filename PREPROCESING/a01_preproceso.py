# %% Librerías 

import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import RandomizedSearchCV

from functions.evaluate_model import evaluate_model

# %% Importacion

df_cnt = pd.read_csv('datasets/contract.csv')
df_prs = pd.read_csv('datasets/personal.csv')
df_int = pd.read_csv('datasets/internet.csv')
df_phn = pd.read_csv('datasets/phone.csv')

# %% Combinacion de librerías

df_full = df_cnt.merge(df_prs, on='customerID', how='left')
df_full = df_full.merge(df_int, on='customerID', how='left')
df_full = df_full.merge(df_phn, on='customerID', how='left')

# %% Creacion de Target

df_full['churn'] = df_full['EndDate'].apply(lambda x: 0 if x == 'No' else 1)

# %% rename

df_full.rename(columns={
    'customerID': 'customer_id',
    'BeginDate': 'begin_date',
    'EndDate': 'end_date',
    'Type': 'contract_type',
    'PaperlessBilling': 'paperless_billing',
    'PaymentMethod': 'payment_method',
    'MonthlyCharges': 'monthly_charges',
    'TotalCharges': 'total_charges',
    'gender': 'gender',
    'SeniorCitizen': 'senior_citizen',
    'Partner': 'partner',
    'Dependents': 'dependents',
    'InternetService': 'internet_service',
    'OnlineSecurity': 'online_security',
    'OnlineBackup': 'online_backup',
    'DeviceProtection': 'device_protection',
    'TechSupport': 'tech_support',
    'StreamingTV': 'streaming_tv',
    'StreamingMovies': 'streaming_movies',
    'MultipleLines': 'multiple_lines'
}, inplace=True)

# %% conversion de tipo de datos y valores nulos

df_full['total_charges'] = pd.to_numeric(df_full['total_charges'], errors='coerce')

df_full['total_charges'].isnull().sum()

df_full = df_full[df_full['total_charges'].notna()]

# %% eliminacion de columnas y codificacion de variables categoricas 

df_model = df_full.drop(columns=[
    'customer_id',    
    'begin_date',     
    'end_date'        
])

df_model = pd.get_dummies(df_model, drop_first=True)

# %% escalado de datos

scaler = StandardScaler()

num_cols = ['monthly_charges', 'total_charges']

df_model[num_cols] = scaler.fit_transform(df_model[num_cols])

# %% segmentacion de datos

X = df_model.drop(columns='churn')
y = df_model['churn']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)


# %% MODELO DUMMY

dummy = DummyClassifier(strategy="most_frequent", random_state=42)
dummy.fit(X_train, y_train)

print("Evaluación del modelo Dummy:")
evaluate_model(dummy, X_val, y_val)

# %% REGRESION LOGISTICA

logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)

print("Evaluación del modelo de Regresión Logística:")
evaluate_model(logreg, X_val, y_val)

# %% RANDOM FOREST

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train, y_train)

print("Evaluación del modelo Random Forest:")
evaluate_model(rf, X_val, y_val)

# %% XGBOOST

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'scale_pos_weight': [1, 2, 3],  
}

xgb_base = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

xgb_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist,
    n_iter=30,  # Número de combinaciones a probar
    scoring='roc_auc',  # Usamos AUC como criterio principal
    cv=3,  # Validación cruzada
    verbose=1,
    random_state=42,
    n_jobs=-1
)

xgb_search.fit(X_train, y_train)

best_xgb = xgb_search.best_estimator_

print("Evaluación del mejor modelo XGBoost (tuned):")
evaluate_model(best_xgb, X_val, y_val)

# %% LIGTHGBM

param_dist_lgbm = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'min_child_samples': [10, 20, 30],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'class_weight': ['balanced', None]
}

lgbm_base = LGBMClassifier(random_state=42)

lgbm_search = RandomizedSearchCV(
    estimator=lgbm_base,
    param_distributions=param_dist_lgbm,
    n_iter=30,
    scoring='roc_auc',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

lgbm_search.fit(X_train, y_train)

best_lgbm = lgbm_search.best_estimator_
print("Evaluación del mejor modelo LightGBM (tuned):")
evaluate_model(best_lgbm, X_val, y_val)

# %% CATBOOST

param_dist_cat = {
    'iterations': [200, 300, 500],
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'border_count': [32, 64, 128],  
    'scale_pos_weight': [1, 2, 3]  
}


cat_base = CatBoostClassifier(
    verbose=0,
    random_state=42
)

cat_search = RandomizedSearchCV(
    estimator=cat_base,
    param_distributions=param_dist_cat,
    n_iter=30,
    scoring='roc_auc',
    cv=3,
    random_state=42,
    verbose=1,
    n_jobs=-1
)

cat_search.fit(X_train, y_train)

best_cat = cat_search.best_estimator_
print("Evaluación del mejor modelo CatBoost (tuned):")
evaluate_model(best_cat, X_val, y_val)

# %% evaluacion de modelos

print("📊 Evaluación final del modelo XGBoost (Test Set):")
evaluate_model(best_xgb, X_test, y_test)

print("\n📊 Evaluación final del modelo CatBoost (Test Set):")
evaluate_model(best_cat, X_test, y_test)





