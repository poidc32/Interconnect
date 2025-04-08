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