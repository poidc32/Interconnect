# %% REGRESION LOGISTICA

logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)

print("Evaluación del modelo de Regresión Logística:")
evaluate_model(logreg, X_val, y_val)