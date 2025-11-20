"""
Transfer Learning with Preprocessed Datasets
Students Performance (preprocessed) → Student Graduation (preprocessed)

Both datasets now have matching preprocessing:
- One-hot encoding on categorical features
- MinMaxScaler on numeric features  
- Rare category removal
- Class balancing via oversampling
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


def load_preprocessed_data():
    """Load preprocessed datasets"""
    print("=" * 70)
    print("TRANSFER LEARNING WITH PREPROCESSED DATASETS")
    print("Students Performance → Student Graduation")
    print("=" * 70)
    
    # Load preprocessed Students Performance (source)
    df_perf_train = pd.read_csv('student-performance-processed/train.csv')
    df_perf_test = pd.read_csv('student-performance-processed/test.csv')
    
    # Load preprocessed Student Graduation (target)
    df_grad_train = pd.read_csv('student-graduation/processed/train.csv')
    df_grad_test = pd.read_csv('student-graduation/processed/test.csv')
    
    print(f"\n✅ Preprocessed data loaded:")
    print(f"   Performance train: {df_perf_train.shape}")
    print(f"   Performance test: {df_perf_test.shape}")
    print(f"   Graduation train: {df_grad_train.shape}")
    print(f"   Graduation test: {df_grad_test.shape}")
    
    return df_perf_train, df_perf_test, df_grad_train, df_grad_test


def align_features(df_source_train, df_source_test, df_target_train, df_target_test):
    """
    Align features between source and target datasets.
    Use only features that exist in both datasets.
    """
    # Get feature columns (exclude Target)
    source_features = [col for col in df_source_train.columns if col != 'Target']
    target_features = [col for col in df_target_train.columns if col != 'Target']
    
    # Find common features
    common_features = list(set(source_features) & set(target_features))
    common_features.sort()
    
    print(f"\n✅ Feature alignment:")
    print(f"   Source features: {len(source_features)}")
    print(f"   Target features: {len(target_features)}")
    print(f"   Common features: {len(common_features)}")
    
    if len(common_features) > 0:
        print(f"   Common features: {common_features[:10]}..." if len(common_features) > 10 else f"   Common features: {common_features}")
    
    # Select common features + Target
    X_source_train = df_source_train[common_features]
    y_source_train = df_source_train['Target']
    
    X_source_test = df_source_test[common_features]
    y_source_test = df_source_test['Target']
    
    X_target_train = df_target_train[common_features]
    y_target_train = df_target_train['Target']
    
    X_target_test = df_target_test[common_features]
    y_target_test = df_target_test['Target']
    
    print(f"\n✅ Aligned dataset shapes:")
    print(f"   Source train: {X_source_train.shape}")
    print(f"   Source test: {X_source_test.shape}")
    print(f"   Target train: {X_target_train.shape}")
    print(f"   Target test: {X_target_test.shape}")
    
    return (X_source_train, y_source_train, X_source_test, y_source_test,
            X_target_train, y_target_train, X_target_test, y_target_test,
            common_features)


def encode_targets(y_source_train, y_source_test, y_target_train, y_target_test):
    """Encode string targets to integers"""
    target_mapping = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
    
    y_source_train_enc = y_source_train.map(target_mapping).values
    y_source_test_enc = y_source_test.map(target_mapping).values
    y_target_train_enc = y_target_train.map(target_mapping).values
    y_target_test_enc = y_target_test.map(target_mapping).values
    
    return y_source_train_enc, y_source_test_enc, y_target_train_enc, y_target_test_enc


def pretrain_model(X_train, y_train, dataset_name="source"):
    """Phase 1: Pretrain on source dataset"""
    print("\n" + "=" * 70)
    print(f"PHASE 1: PRETRAINING ON {dataset_name.upper()}")
    print("=" * 70)
    
    n_features = X_train.shape[1]
    
    # Adjust architecture based on number of features
    if n_features < 10:
        hidden_layers = (64, 32)
    elif n_features < 30:
        hidden_layers = (128, 64)
    else:
        hidden_layers = (128, 64, 32)
    
    print(f"\n📊 Training on {X_train.shape[0]} samples with {n_features} features")
    print(f"   Architecture: {hidden_layers}")
    
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        random_state=42,
        verbose=True
    )
    
    mlp.fit(X_train, y_train)
    
    train_acc = mlp.score(X_train, y_train)
    print(f"\n✅ Pretraining complete!")
    print(f"   Training accuracy: {train_acc:.4f}")
    
    return mlp


def finetune_model(pretrained_mlp, X_train, y_train):
    """Phase 2: Finetune on target dataset with transferred weights"""
    print("\n" + "=" * 70)
    print("PHASE 2: FINETUNING ON TARGET DATASET")
    print("=" * 70)
    
    hidden_layers = pretrained_mlp.hidden_layer_sizes
    
    # Create new model with same architecture
    finetune_mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        learning_rate_init=0.0001,  # Lower LR for finetuning
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=15,
        warm_start=False,
        random_state=42,
        verbose=True
    )
    
    # Initialize with small sample
    from sklearn.model_selection import train_test_split
    X_init, _, y_init, _ = train_test_split(
        X_train, y_train,
        train_size=min(500, len(X_train) // 2),
        random_state=42,
        stratify=y_train
    )
    
    print(f"\n📊 Initializing model...")
    finetune_mlp.fit(X_init, y_init)
    
    print(f"\n📊 Transferring pretrained weights...")
    
    # Copy weights from pretrained model
    for i in range(len(pretrained_mlp.coefs_)):
        if pretrained_mlp.coefs_[i].shape == finetune_mlp.coefs_[i].shape:
            finetune_mlp.coefs_[i][:] = pretrained_mlp.coefs_[i]
            finetune_mlp.intercepts_[i][:] = pretrained_mlp.intercepts_[i]
    
    # Reset counters and enable warm_start
    finetune_mlp.n_iter_ = 0
    finetune_mlp.t_ = 0
    finetune_mlp.warm_start = True
    
    print(f"📊 Finetuning on {X_train.shape[0]} samples with pretrained weights")
    finetune_mlp.fit(X_train, y_train)
    
    train_acc = finetune_mlp.score(X_train, y_train)
    print(f"\n✅ Finetuning complete!")
    print(f"   Training accuracy: {train_acc:.4f}")
    
    return finetune_mlp


def train_baseline(X_train, y_train):
    """Train baseline model without transfer learning"""
    print("\n" + "=" * 70)
    print("BASELINE: TRAINING WITHOUT TRANSFER LEARNING")
    print("=" * 70)
    
    n_features = X_train.shape[1]
    
    if n_features < 10:
        hidden_layers = (64, 32)
    elif n_features < 30:
        hidden_layers = (128, 64)
    else:
        hidden_layers = (128, 64, 32)
    
    baseline_mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        random_state=42,
        verbose=True
    )
    
    print(f"\n📊 Training on {X_train.shape[0]} samples with {n_features} features")
    print(f"   Architecture: {hidden_layers}")
    
    baseline_mlp.fit(X_train, y_train)
    
    train_acc = baseline_mlp.score(X_train, y_train)
    print(f"\n✅ Baseline training complete!")
    print(f"   Training accuracy: {train_acc:.4f}")
    
    return baseline_mlp


def evaluate_models(transfer_model, baseline_model, X_test, y_test):
    """Evaluate and compare both models"""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    # Predictions
    y_pred_transfer = transfer_model.predict(X_test)
    y_pred_baseline = baseline_model.predict(X_test)
    
    # Metrics
    acc_transfer = accuracy_score(y_test, y_pred_transfer)
    acc_baseline = accuracy_score(y_test, y_pred_baseline)
    
    bal_acc_transfer = balanced_accuracy_score(y_test, y_pred_transfer)
    bal_acc_baseline = balanced_accuracy_score(y_test, y_pred_baseline)
    
    print(f"\n📊 TRANSFER LEARNING MODEL:")
    print(f"   Accuracy:          {acc_transfer:.4f}")
    print(f"   Balanced Accuracy: {bal_acc_transfer:.4f}")
    
    print(f"\n📊 BASELINE MODEL:")
    print(f"   Accuracy:          {acc_baseline:.4f}")
    print(f"   Balanced Accuracy: {bal_acc_baseline:.4f}")
    
    improvement_acc = acc_transfer - acc_baseline
    improvement_bal = bal_acc_transfer - bal_acc_baseline
    
    print(f"\n📈 IMPROVEMENT FROM TRANSFER LEARNING:")
    print(f"   Accuracy:          {improvement_acc:+.4f} ({improvement_acc/acc_baseline*100:+.2f}%)")
    print(f"   Balanced Accuracy: {improvement_bal:+.4f} ({improvement_bal/bal_acc_baseline*100:+.2f}%)")
    
    # Classification reports
    target_names = ['Dropout', 'Enrolled', 'Graduate']
    
    print("\n" + "=" * 70)
    print("TRANSFER LEARNING - CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(y_test, y_pred_transfer, target_names=target_names))
    
    print("\n" + "=" * 70)
    print("BASELINE - CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(y_test, y_pred_baseline, target_names=target_names))
    
    return {
        'transfer': {'accuracy': acc_transfer, 'balanced_accuracy': bal_acc_transfer},
        'baseline': {'accuracy': acc_baseline, 'balanced_accuracy': bal_acc_baseline}
    }


def main():
    """Main transfer learning pipeline"""
    
    # Step 1: Load preprocessed data
    df_perf_train, df_perf_test, df_grad_train, df_grad_test = load_preprocessed_data()
    
    # Step 2: Align features
    (X_source_train, y_source_train, X_source_test, y_source_test,
     X_target_train, y_target_train, X_target_test, y_target_test,
     common_features) = align_features(df_perf_train, df_perf_test, df_grad_train, df_grad_test)
    
    if len(common_features) == 0:
        print("\n❌ No common features found! Cannot proceed with transfer learning.")
        return None, None, None
    
    # Step 3: Encode targets
    y_source_train, y_source_test, y_target_train, y_target_test = encode_targets(
        y_source_train, y_source_test, y_target_train, y_target_test
    )
    
    # Step 4: Pretrain on source dataset
    pretrained_model = pretrain_model(X_source_train, y_source_train, "Students Performance")
    
    # Evaluate on source test set
    source_test_acc = pretrained_model.score(X_source_test, y_source_test)
    print(f"\n📊 Source test accuracy: {source_test_acc:.4f}")
    
    # Step 5: Finetune on target dataset
    transfer_model = finetune_model(pretrained_model, X_target_train, y_target_train)
    
    # Step 6: Train baseline on target dataset
    baseline_model = train_baseline(X_target_train, y_target_train)
    
    # Step 7: Evaluate on target test set
    results = evaluate_models(transfer_model, baseline_model, X_target_test, y_target_test)
    
    print("\n" + "=" * 70)
    print("✅ TRANSFER LEARNING PIPELINE COMPLETE!")
    print("=" * 70)
    
    return transfer_model, baseline_model, results


if __name__ == "__main__":
    transfer_model, baseline_model, results = main()
