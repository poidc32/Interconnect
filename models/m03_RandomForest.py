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