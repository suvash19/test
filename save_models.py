import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Define base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('ada', AdaBoostClassifier(n_estimators=100, random_state=42)), 
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
]

# Create stacking classifier
stack = StackingClassifier(
    estimators=base_models,
    final_estimator=RandomForestClassifier(n_estimators=200, random_state=42),
    cv=5,
    n_jobs=-1
)

# Save model
def save_model(model, filename):
    try:
        joblib.dump(model, filename)
        print(f"Model saved successfully as {filename}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

# Save both base models and stacked model
for name, model in base_models:
    save_model(model, f'models/{name}_model.pkl')
    
save_model(stack, 'models/stacked_model.pkl')