"""
Transfer Learning: Students Performance → Student Graduation
Using MLPClassifier from pipeline.ipynb
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data():
    """Load both datasets"""
    
    # Load Students Performance (source)
    df_performance = pd.read_csv('Students Performance Dataset (1).csv')
    
    # Load Student Graduation (target)
    df_train = pd.read_csv('./student-graduation/processed/train.csv')
    df_test = pd.read_csv('./student-graduation/processed/test.csv')
    
    # Encode graduation target
    le = LabelEncoder()
    df_train['Target'] = le.fit_transform(df_train['Target'])
    df_test['Target'] = le.transform(df_test['Target'])
    
    print("✅ Data loaded successfully")
    print(f"   Performance dataset: {df_performance.shape}")
    print(f"   Graduation train: {df_train.shape}")
    print(f"   Graduation test: {df_test.shape}")
    
    return df_performance, df_train, df_test, le


def prepare_performance_dataset(df_performance):
    """Transform Students Performance to match Graduation schema"""
    
    # Map Grade to 3-class Target (0=Dropout, 1=Enrolled, 2=Graduate)
    grade_map = {
        'A': 2, 'B': 2, 'C': 2,  # Graduate
        'D': 1, 'E': 1,           # Enrolled
        'F': 0                    # Dropout
    }
    df_performance['Target'] = df_performance['Grade'].map(grade_map).fillna(1)
    
    # Convert Gender to binary
    df_performance['Gender'] = df_performance['Gender'].map({'Male': 1, 'Female': 0})
    
    # Select numeric columns that might align
    numeric_cols = df_performance.select_dtypes(include=[np.number]).columns.tolist()
    
    # Keep Target and numeric features
    df_clean = df_performance[numeric_cols].copy()
    if 'Target' not in df_clean.columns:
        df_clean['Target'] = df_performance['Target']
    
    print(f"\n✅ Performance dataset prepared")
    print(f"   Shape: {df_clean.shape}")
    print(f"   Target distribution: {df_clean['Target'].value_counts().to_dict()}")
    
    return df_clean


def align_features(df_source, df_target_train):
    """Align source dataset features to match target dataset"""
    
    target_features = [col for col in df_target_train.columns if col != 'Target']
    
    # Add missing columns as zeros
    for col in target_features:
        if col not in df_source.columns:
            df_source[col] = 0
    
    # Keep only target features + Target
    df_aligned = df_source[target_features + ['Target']].copy()
    
    print(f"\n✅ Features aligned")
    print(f"   Source shape: {df_aligned.shape}")
    print(f"   Target shape: {df_target_train.shape}")
    
    return df_aligned


def pretrain_model(X_source, y_source):
    """Phase 1: Pretrain on Students Performance dataset"""
    
    print("\n" + "="*70)
    print("PHASE 1: PRETRAINING ON STUDENTS PERFORMANCE DATASET")
    print("="*70)
    
    # Create preprocessing + model pipeline
    pretrain_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            random_state=42,
            verbose=True
        ))
    ])
    
    print(f"\n📊 Training on {X_source.shape[0]} samples with {X_source.shape[1]} features")
    pretrain_pipeline.fit(X_source, y_source)
    
    train_score = pretrain_pipeline.score(X_source, y_source)
    print(f"\n✅ Pretraining complete! Training accuracy: {train_score:.4f}")
    
    return pretrain_pipeline


def finetune_model(pretrain_pipeline, X_target, y_target):
    """Phase 2: Finetune on Student Graduation dataset"""
    
    print("\n" + "="*70)
    print("PHASE 2: FINETUNING ON STUDENT GRADUATION DATASET")
    print("="*70)
    
    # Extract pretrained components
    pretrained_mlp = pretrain_pipeline.named_steps['mlp']
    
    # Transform data using pretrained preprocessing
    X_transformed = pretrain_pipeline.named_steps['imputer'].transform(X_target)
    X_transformed = pretrain_pipeline.named_steps['scaler'].transform(X_transformed)
    
    # Create new MLP with lower learning rate
    finetune_mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        learning_rate_init=0.0001,  # Lower learning rate for finetuning
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=15,
        random_state=42,
        warm_start=True,
        verbose=True
    )
    
    # First fit to initialize structure
    finetune_mlp.fit(X_transformed, y_target)
    
    print(f"\n📊 Loading pretrained weights and continuing training...")
    
    # Copy pretrained weights AFTER first fit
    for i in range(len(pretrained_mlp.coefs_)):
        finetune_mlp.coefs_[i][:] = pretrained_mlp.coefs_[i]
        finetune_mlp.intercepts_[i][:] = pretrained_mlp.intercepts_[i]
    
    # Reset iteration counter but keep weights
    finetune_mlp.n_iter_ = 0
    finetune_mlp._no_improvement_count = 0
    finetune_mlp.best_loss_ = np.inf
    
    print(f"📊 Finetuning on {X_target.shape[0]} samples with pretrained weights")
    
    # Continue training with warm_start from pretrained weights
    finetune_mlp.fit(X_transformed, y_target)
    
    # Create final pipeline with finetuned model
    finetune_pipeline = Pipeline([
        ('imputer', pretrain_pipeline.named_steps['imputer']),
        ('scaler', pretrain_pipeline.named_steps['scaler']),
        ('mlp', finetune_mlp)
    ])
    
    train_score = finetune_pipeline.score(X_target, y_target)
    print(f"\n✅ Finetuning complete! Training accuracy: {train_score:.4f}")
    
    return finetune_pipeline


def train_baseline(X_train, y_train):
    """Train baseline model without transfer learning"""
    
    print("\n" + "="*70)
    print("BASELINE: TRAINING WITHOUT TRANSFER LEARNING")
    print("="*70)
    
    baseline_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            random_state=42,
            verbose=True
        ))
    ])
    
    print(f"\n📊 Training on {X_train.shape[0]} samples")
    baseline_pipeline.fit(X_train, y_train)
    
    train_score = baseline_pipeline.score(X_train, y_train)
    print(f"\n✅ Baseline training complete! Training accuracy: {train_score:.4f}")
    
    return baseline_pipeline


def evaluate_models(transfer_model, baseline_model, X_test, y_test):
    """Compare transfer learning vs baseline"""
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    # Transfer learning predictions
    y_pred_transfer = transfer_model.predict(X_test)
    acc_transfer = accuracy_score(y_test, y_pred_transfer)
    bal_acc_transfer = balanced_accuracy_score(y_test, y_pred_transfer)
    
    # Baseline predictions
    y_pred_baseline = baseline_model.predict(X_test)
    acc_baseline = accuracy_score(y_test, y_pred_baseline)
    bal_acc_baseline = balanced_accuracy_score(y_test, y_pred_baseline)
    
    print("\n📊 TRANSFER LEARNING MODEL:")
    print(f"   Accuracy:          {acc_transfer:.4f}")
    print(f"   Balanced Accuracy: {bal_acc_transfer:.4f}")
    
    print("\n📊 BASELINE MODEL:")
    print(f"   Accuracy:          {acc_baseline:.4f}")
    print(f"   Balanced Accuracy: {bal_acc_baseline:.4f}")
    
    print("\n📈 IMPROVEMENT:")
    print(f"   Accuracy:          {acc_transfer - acc_baseline:+.4f}")
    print(f"   Balanced Accuracy: {bal_acc_transfer - bal_acc_baseline:+.4f}")
    
    print("\n" + "="*70)
    print("TRANSFER LEARNING CLASSIFICATION REPORT:")
    print("="*70)
    print(classification_report(y_test, y_pred_transfer, target_names=['Dropout', 'Enrolled', 'Graduate']))
    
    print("\n" + "="*70)
    print("BASELINE CLASSIFICATION REPORT:")
    print("="*70)
    print(classification_report(y_test, y_pred_baseline, target_names=['Dropout', 'Enrolled', 'Graduate']))
    
    return {
        'transfer': {'accuracy': acc_transfer, 'balanced_accuracy': bal_acc_transfer},
        'baseline': {'accuracy': acc_baseline, 'balanced_accuracy': bal_acc_baseline}
    }


def main():
    """Main transfer learning pipeline"""
    
    print("\n" + "="*70)
    print("TRANSFER LEARNING PIPELINE")
    print("Students Performance → Student Graduation")
    print("="*70)
    
    # Step 1: Load data
    df_performance, df_train, df_test, le = load_and_prepare_data()
    
    # Step 2: Prepare source dataset
    df_performance_clean = prepare_performance_dataset(df_performance)
    
    # Step 3: Align features
    df_source_aligned = align_features(df_performance_clean, df_train)
    
    # Step 4: Prepare train/test splits
    X_source = df_source_aligned.drop(columns=['Target'])
    y_source = df_source_aligned['Target']
    
    X_train = df_train.drop(columns=['Target'])
    y_train = df_train['Target']
    
    X_test = df_test.drop(columns=['Target'])
    y_test = df_test['Target']
    
    # Step 5: Pretrain on source dataset
    pretrained_model = pretrain_model(X_source, y_source)
    
    # Step 6: Finetune on target dataset
    transfer_model = finetune_model(pretrained_model, X_train, y_train)
    
    # Step 7: Train baseline
    baseline_model = train_baseline(X_train, y_train)
    
    # Step 8: Evaluate and compare
    results = evaluate_models(transfer_model, baseline_model, X_test, y_test)
    
    print("\n✅ Pipeline completed successfully!")
    
    return transfer_model, baseline_model, results


if __name__ == "__main__":
    transfer_model, baseline_model, results = main()
