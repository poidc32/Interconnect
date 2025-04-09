from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from functions.evaluate_model import evaluate_model

# %% MODELO DUMMY

# TODO: Leer X_train, y_train, X_val, y_val

dummy = DummyClassifier(strategy="most_frequent", random_state=42)
dummy.fit(X_train, y_train)

print("Evaluación del modelo Dummy:")
evaluate_model(dummy, X_val, y_val)

# %% REGRESION LOGISTICA

logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)

print("Evaluación del modelo de Regresión Logística:")
evaluate_model(logreg, X_val, y_val)