from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def train_logistic(X_train, y_train):
    model = LogisticRegression(
        C=1000, 
        max_iter=3000, 
        penalty='l1',
        random_state=42,
        solver='liblinear')
    model.fit(X_train, y_train)
    return model

def train_tree(X_train, y_train):
    model = DecisionTreeClassifier(
        ccp_alpha=6.639769763389794e-05,
        random_state=42)
    model.fit(X_train, y_train)
    return model

def train_rf(X_train, y_train):
    model = RandomForestClassifier(
        max_depth=30, min_samples_split=20, 
        n_estimators=500,
        random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgb(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=0.5,
        reg_alpha=0.5,
        reg_lambda=5,
        random_state=42,
        eval_metric='auc'
    )
    model.fit(X_train, y_train)
    return model