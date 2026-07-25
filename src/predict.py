def predict(model, X_val):
    y_pred = model.predict_proba(X_val)
    return y_pred