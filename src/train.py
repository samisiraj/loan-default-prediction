from sklearn.linear_model import LogisticRegression


def train_logistic(X_train, y_train):
    model = LogisticRegression(C=1000, max_iter=3000, penalty='l1',
                   random_state=42, solver='liblinear')
    model.fit(X_train, y_train)
    return model