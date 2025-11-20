"""
Transfer Learning: Students Performance → Student Graduation
Using ONLY overlapping features for better transfer
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


def find_overlapping_features(df_source, df_target):
    """Find features that exist in both datasets"""
    
    # Map Grade to Target in source
    grade_map = {
        'A': 2, 'B': 2, 'C': 2,  # Graduate
        'D': 1, 'E': 1,           # Enrolled
        'F': 0                    # Dropout
    }
    df_source['Target'] = df_source['Grade'].map(grade_map).fillna(1)
    
    # Rename Age to match target dataset
    if 'Age' in df_source.columns and 'Age at enrollment' in df_target.columns:
        df_source = df_source.rename(columns={'Age': 'Age at enrollment'})
    
    # Get numeric columns from source (excluding Target)
    source_numeric = [col for col in df_source.select_dtypes(include=[np.number]).columns.tolist() 
                      if col != 'Target']
    
    # Get features from target (excluding Target)
    target_features = [col for col in df_target.columns if col != 'Target']
    
    # Find common features - only exact name matches
    common_features = []
    for feat in source_numeric:
        if feat in target_features:
            common_features.append(feat)
    
    print(f"\n✅ Feature overlap analysis:")
    print(f"   Source numeric features: {len(source_numeric)}")
    print(f"   Target features: {len(target_features)}")
    print(f"   Overlapping features: {len(common_features)}")
    if len(common_features) > 0:
        print(f"   Common features: {common_features}")
    else:
        print(f"   ⚠️  No exact feature name matches found!")
    
    return common_features, df_source


def prepare_datasets_with_overlap(df_source, df_train, df_test, common_features):
    """Prepare datasets using only overlapping features"""
    
    # Map Grade to Target in source
    grade_map = {
        'A': 2, 'B': 2, 'C': 2,  # Graduate
        'D': 1, 'E': 1,           # Enrolled
        'F': 0                    # Dropout
    }
    df_source['Target'] = df_source['Grade'].map(grade_map).fillna(1)
    
    # Convert Gender to binary in source
    if 'Gender' in df_source.columns:
        df_source['Gender'] = df_source['Gender'].map({'Male': 1, 'Female': 0})
    
    # Select only common features + Target
    df_source_overlap = df_source[common_features + ['Target']].copy()
    df_train_overlap = df_train[common_features + ['Target']].copy()
    df_test_overlap = df_test[common_features + ['Target']].copy()
    
    # Remove any NaN values
    df_source_overlap = df_source_overlap.dropna()
    df_train_overlap = df_train_overlap.dropna()
    df_test_overlap = df_test_overlap.dropna()
    
    print(f"\n✅ Datasets prepared with overlapping features:")
    print(f"   Source shape: {df_source_overlap.shape}")
    print(f"   Train shape: {df_train_overlap.shape}")
    print(f"   Test shape: {df_test_overlap.shape}")
    
    return df_source_overlap, df_train_overlap, df_test_overlap


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
    """Main transfer learning pipeline with overlapping features only"""
    
    print("\n" + "="*70)
    print("TRANSFER LEARNING PIPELINE (OVERLAPPING FEATURES ONLY)")
    print("Students Performance → Student Graduation")
    print("="*70)
    
    # Step 1: Load data
    df_performance, df_train, df_test, le = load_and_prepare_data()
    
    # Step 2: Find overlapping features
    common_features, df_performance = find_overlapping_features(df_performance, df_train)
    
    if len(common_features) == 0:
        print("\n❌ No overlapping features found! Cannot proceed with transfer learning.")
        print("\nTrying with common feature names instead...")
        
        # Alternative: Use Age and Gender if they exist
        common_features = []
        if 'Age' in df_performance.columns:
            common_features.append('Age')
        if 'Gender' in df_performance.columns:
            common_features.append('Gender')
        
        # Add any numeric columns that might match
        source_cols = set(df_performance.columns)
        target_cols = set(df_train.columns)
        
        # Look for partial matches
        for s_col in source_cols:
            for t_col in target_cols:
                if s_col.lower() in t_col.lower() or t_col.lower() in s_col.lower():
                    if t_col != 'Target' and t_col not in common_features:
                        common_features.append(t_col)
        
        if len(common_features) == 0:
            print("\n❌ Still no common features. Using numeric features from both datasets.")
            # Use all numeric features from both
            source_numeric = df_performance.select_dtypes(include=[np.number]).columns.tolist()
            target_numeric = df_train.select_dtypes(include=[np.number]).columns.tolist()
            target_numeric = [c for c in target_numeric if c != 'Target']
            
            # Use the minimum set
            common_features = source_numeric[:min(len(source_numeric), len(target_numeric))]
            print(f"\n⚠️  Using first {len(common_features)} numeric features as proxy")
    
    # Step 3: Prepare datasets with overlapping features
    df_source_overlap, df_train_overlap, df_test_overlap = prepare_datasets_with_overlap(
        df_performance, df_train, df_test, common_features
    )
    
    # Step 4: Prepare train/test splits
    X_source = df_source_overlap.drop(columns=['Target'])
    y_source = df_source_overlap['Target']
    
    X_train = df_train_overlap.drop(columns=['Target'])
    y_train = df_train_overlap['Target']
    
    X_test = df_test_overlap.drop(columns=['Target'])
    y_test = df_test_overlap['Target']
    
    print(f"\n✅ Final data splits:")
    print(f"   Source: {X_source.shape[0]} samples, {X_source.shape[1]} features")
    print(f"   Train:  {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"   Test:   {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
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
