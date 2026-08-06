from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def train_logistic(X_train, y_train):
    model = LogisticRegression(C=1000, max_iter=3000, penalty='l1',
                   random_state=42, solver='liblinear')
    model.fit(X_train, y_train)
    return model

def train_tree(X_train, y_train):
    model = DecisionTreeClassifier(ccp_alpha=6.639769763389794e-05,
                       random_state=42)
    model.fit(X_train, y_train)
    return model