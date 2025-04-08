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