"""
Preprocessing for Students Performance Dataset
Matching the preprocessing pipeline used for Student Graduation dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data():
    """Load the Students Performance dataset"""
    df = pd.read_csv('Students Performance Dataset (1).csv')
    print(f"✅ Data loaded: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    return df


def map_target(df):
    """
    Map Grade to Target (3-class problem matching graduation dataset):
    - A, B, C → Graduate
    - D, E → Enrolled  
    - F → Dropout
    """
    grade_mapping = {
        'A': 'Graduate',
        'B': 'Graduate',
        'C': 'Graduate',
        'D': 'Enrolled',
        'E': 'Enrolled',
        'F': 'Dropout'
    }
    df['Target'] = df['Grade'].map(grade_mapping)
    
    print(f"\n✅ Target distribution:")
    print(df['Target'].value_counts())
    
    return df


def identify_categorical_columns(df):
    """
    Identify categorical columns that should be one-hot encoded
    (columns without natural ordering)
    """
    # Categorical columns in student performance that have no natural order
    categorical_cols = [
        'Department',  # CS, Engineering, Business, Mathematics
        'Parent_Education_Level',  # High School, Bachelor's, Master's, PhD
        'Family_Income_Level',  # Low, Medium, High
        'Extracurricular_Activities',  # Yes, No
        'Internet_Access_at_Home'  # Yes, No
    ]
    
    # Gender should be binary encoded (not OHE since only 2 categories)
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    
    print(f"\n✅ Categorical columns for OHE: {categorical_cols}")
    
    return categorical_cols


def one_hot_encode(df, categorical_cols):
    """Apply one-hot encoding to categorical columns"""
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    
    print(f"\n✅ After OHE: {df_encoded.shape}")
    print(f"   New columns: {df_encoded.shape[1] - df.shape[1]} added")
    
    return df_encoded


def train_test_split_data(df):
    """Split data into train and test sets"""
    X = df.drop(columns=['Target', 'Grade', 'Student_ID', 'First_Name', 'Last_Name', 'Email'])
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"\n✅ Train/Test split:")
    print(f"   Train: {X_train.shape}")
    print(f"   Test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def scale_numeric_features(X_train, X_test):
    """Scale numeric features using MinMaxScaler (matching graduation preprocessing)"""
    
    # Identify numeric columns (excluding binary columns from OHE)
    numeric_cols = [
        'Age',
        'Attendance (%)',
        'Midterm_Score',
        'Final_Score',
        'Assignments_Avg',
        'Quizzes_Avg',
        'Participation_Score',
        'Projects_Score',
        'Total_Score',
        'Study_Hours_per_Week',
        'Stress_Level (1-10)',
        'Sleep_Hours_per_Night',
        'Gender'  # Already encoded as 0/1
    ]
    
    # Only scale columns that exist
    numeric_cols = [col for col in numeric_cols if col in X_train.columns]
    
    scaler = MinMaxScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    print(f"\n✅ Scaled {len(numeric_cols)} numeric columns")
    
    return X_train_scaled, X_test_scaled, scaler


def remove_rare_categories(X_train, X_test, categorical_cols, threshold=0.03):
    """
    Remove one-hot encoded columns that appear in less than threshold% of samples
    (matching graduation preprocessing)
    """
    min_count = int(len(X_train) * threshold)
    
    # Find all OHE columns
    ohe_columns = []
    for base in categorical_cols:
        prefix = base + '_'
        ohe_columns.extend([c for c in X_train.columns if c.startswith(prefix)])
    
    # Calculate frequency
    freq = X_train[ohe_columns].sum(axis=0)
    
    # Find columns to drop
    cols_to_drop = freq[freq < min_count].index.tolist()
    
    # Drop from both train and test
    X_train_filtered = X_train.drop(columns=cols_to_drop)
    X_test_filtered = X_test.drop(columns=cols_to_drop)
    
    print(f"\n✅ Rare category filtering (threshold={threshold*100}%):")
    print(f"   Columns dropped: {len(cols_to_drop)}")
    print(f"   Remaining shape - Train: {X_train_filtered.shape}, Test: {X_test_filtered.shape}")
    
    return X_train_filtered, X_test_filtered


def oversample_minority_classes(X_train, y_train):
    """
    Oversample minority classes to balance the dataset
    (matching graduation preprocessing)
    """
    train_df = X_train.copy()
    train_df['Target'] = y_train.values
    
    # Separate by class
    train_graduate = train_df[train_df['Target'] == 'Graduate']
    train_enrolled = train_df[train_df['Target'] == 'Enrolled']
    train_dropout = train_df[train_df['Target'] == 'Dropout']
    
    print(f"\n✅ Class distribution before oversampling:")
    print(f"   Graduate: {len(train_graduate)}")
    print(f"   Enrolled: {len(train_enrolled)}")
    print(f"   Dropout: {len(train_dropout)}")
    
    # Find majority class size
    max_size = max(len(train_graduate), len(train_enrolled), len(train_dropout))
    
    # Oversample to match majority class
    train_graduate_balanced = train_graduate.sample(n=max_size, replace=True, random_state=42)
    train_enrolled_balanced = train_enrolled.sample(n=max_size, replace=True, random_state=42)
    train_dropout_balanced = train_dropout.sample(n=max_size, replace=True, random_state=42)
    
    # Concatenate
    train_balanced = pd.concat([train_graduate_balanced, train_enrolled_balanced, train_dropout_balanced])
    
    # Shuffle
    train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n✅ Class distribution after oversampling:")
    print(f"   Graduate: {len(train_graduate_balanced)}")
    print(f"   Enrolled: {len(train_enrolled_balanced)}")
    print(f"   Dropout: {len(train_dropout_balanced)}")
    print(f"   Total: {len(train_balanced)}")
    
    return train_balanced


def save_processed_data(train_df, test_df):
    """Save processed data to CSV files"""
    train_df.to_csv('student-performance-processed/train.csv', index=False)
    test_df.to_csv('student-performance-processed/test.csv', index=False)
    
    print(f"\n✅ Saved processed data:")
    print(f"   Train: student-performance-processed/train.csv")
    print(f"   Test: student-performance-processed/test.csv")


def main():
    """Main preprocessing pipeline"""
    print("=" * 70)
    print("PREPROCESSING PIPELINE FOR STUDENTS PERFORMANCE DATASET")
    print("Matching Student Graduation preprocessing steps")
    print("=" * 70)
    
    # Step 1: Load data
    df = load_data()
    
    # Step 2: Map Grade to Target
    df = map_target(df)
    
    # Step 3: Identify categorical columns
    categorical_cols = identify_categorical_columns(df)
    
    # Step 4: One-hot encode categorical columns
    df_encoded = one_hot_encode(df, categorical_cols)
    
    # Step 5: Train/test split
    X_train, X_test, y_train, y_test = train_test_split_data(df_encoded)
    
    # Step 6: Scale numeric features
    X_train_scaled, X_test_scaled, scaler = scale_numeric_features(X_train, X_test)
    
    # Step 7: Remove rare categories
    X_train_filtered, X_test_filtered = remove_rare_categories(
        X_train_scaled, X_test_scaled, categorical_cols, threshold=0.03
    )
    
    # Step 8: Oversample minority classes (only on train)
    train_balanced = oversample_minority_classes(X_train_filtered, y_train)
    
    # Step 9: Prepare test dataframe
    test_df = X_test_filtered.copy()
    test_df['Target'] = y_test.values
    
    # Step 10: Save processed data
    import os
    os.makedirs('student-performance-processed', exist_ok=True)
    save_processed_data(train_balanced, test_df)
    
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\nFinal shapes:")
    print(f"  Train (balanced): {train_balanced.shape}")
    print(f"  Test: {test_df.shape}")
    print(f"\nFeatures: {train_balanced.shape[1] - 1} (excluding Target)")
    
    return train_balanced, test_df


if __name__ == "__main__":
    train_df, test_df = main()
