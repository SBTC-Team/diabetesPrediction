import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

def main():
    # Paths
    dataset_path = r'c:\Users\navi\Kaggle\archive\diabetes_dataset.csv'
    test_path = r'c:\Users\navi\Kaggle\archive\test.csv'
    output_path = r'c:\Users\navi\Kaggle\archive\predictions.csv'

    print("Loading data...")
    df_train = pd.read_csv(dataset_path)
    df_test = pd.read_csv(test_path)

    # Intersection of columns (excluding target and ID)
    target_col = 'diagnosed_diabetes'
    id_col = 'id'
    
    # Columns in test.csv
    test_cols = set(df_test.columns)
    # Columns in diabetes_dataset.csv
    train_cols = set(df_train.columns)
    
    # Common features
    features = list(test_cols.intersection(train_cols) - {id_col})
    
    print(f"Features used: {features}")

    X_train = df_train[features]
    y_train = df_train[target_col]
    X_test_final = df_test[features]
    test_ids = df_test[id_col]

    # Identify categorical and numerical columns
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_train.select_dtypes(exclude=['object', 'category']).columns.tolist()

    print(f"Categorical columns: {cat_cols}")
    print(f"Numerical columns: {num_cols}")

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
        ])

    # XGBoost classifier
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
    ])

    print("Training XGBoost model...")
    clf.fit(X_train, y_train)

    print("Generating predictions...")
    # predict_proba returns [prob_0, prob_1]
    probs = clf.predict_proba(X_test_final)[:, 1]

    # Create output dataframe
    results = pd.DataFrame({
        'id': test_ids,
        'diagnosed_diabetes': np.round(probs, 2)
    })

    print(f"Saving results to {output_path}...")
    results.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
