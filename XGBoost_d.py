import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sns.set()


dataset_path = r'c:\Users\navi\Kaggle\archive\diabetes_dataset.csv'
test_path = r'c:\Users\navi\Kaggle\archive\test.csv'
output_path = r'c:\Users\navi\Kaggle\archive\predictionsXGB.csv'
df = pd.read_csv(dataset_path)

cols_to_drop = [
    'gender', 'ethnicity', 'education_level', 'income_level', 
    'employment_status', 'smoking_status', 'diabetes_stage',
    'family_history_diabetes', 'diagnosed_diabetes' # The target
]

X = df.drop(columns=cols_to_drop)
y = df['diagnosed_diabetes']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

 
# XGBoost handles the full dataset much better than SVM
print("Training XGBoost model...")
# X_train_subset = X_train_scaled[:10000]
# y_train_subset = y_train[:10000]

xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train)


y_pred = xgb_model.predict(X_test_scaled)


print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - XGBoost Diabetes Prediction')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()