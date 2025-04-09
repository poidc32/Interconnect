import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

# %% procedimiento de evaluación

def evaluate_model(model, X_val, y_val, show_cm=True):
    
    y_pred = model.predict(X_val)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_val)[:,1]
    else:
        y_prob = model.decision_function(X_val)

    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")

    if show_cm:
        cm = confusion_matrix(y_val, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title("Matriz de Confusión")
        plt.show()


print(f"El nombre de esta sección es: {__name__}")

if __name__ == "__main__":
    # Este bloque se ejecutará solo si el script se ejecuta directamente
    print("Este script no está diseñado para ejecutarse directamente.")
    print("Por favor, importa la función evaluate_model desde otro script.")


