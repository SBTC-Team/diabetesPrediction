import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

def main():
    # Paths
    dataset_path = r'c:\Users\sbtc\diabetes\diabetes_dataset.csv'
    test_path = r'c:\Users\sbtc\diabetes\test.csv'
    output_path = r'c:\Users\sbtc\diabetes\predictions.csv'

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

    # SVM model with probability=True
    # Using a subset for training if dataset is too large, but user asked to use the model.
    # The existing model in support_vect_d.py used a subset of 10,000.
    # Let's try to use more if possible, but for SVM, 100k+ rows might be slow.
    # Let's see how many rows we have.
    print(f"Training set rows: {len(df_train)}")
    
    # To avoid taking forever, I'll use a subset if it's very large, 
    # but the user said "que el modelo tome las columnas posibles... para entrenarse".
    # I will use a reasonable subset or the full data if it's not too big.
    # SVC scale is O(n^2) to O(n^3).
    
    train_size = min(len(df_train), 20000) # Increased to 20k for better accuracy while keeping it relatively fast
    print(f"Using a subset of {train_size} samples for SVM training...")
    
    X_train_sub = X_train.sample(train_size, random_state=42)
    y_train_sub = y_train.loc[X_train_sub.index]

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(kernel='rbf', probability=True, random_state=42))
    ])

    print("Training SVM model...")
    clf.fit(X_train_sub, y_train_sub)

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
