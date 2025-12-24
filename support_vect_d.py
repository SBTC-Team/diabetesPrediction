import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sns.set()


ruta = r'C:\Users\navi\Kaggle\archive\diabetes_dataset.csv'
df = pd.read_csv(ruta)

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

 
# 10,000 samples
print("Training SVM model (using a subset of 10,000 samples for speed)...")
X_train_subset = X_train_scaled[:10000]
y_train_subset = y_train[:10000]

svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
svm_model.fit(X_train_subset, y_train_subset)


y_pred = svm_model.predict(X_test_scaled)


print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - SVM Diabetes Prediction')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()