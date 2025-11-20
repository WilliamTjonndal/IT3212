"""
Transfer Learning with Semantic Feature Mapping
Students Performance → Student Graduation

This implementation creates meaningful feature mappings between datasets:
- Parent_Education_Level → Mother's/Father's qualification
- Stress_Level → Displaced (high stress = displaced)
- Gender → Gender
- Department → Course (semantic mapping)
- Age → Age at enrollment
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# def map_department_to_course(department):
#     """
#     Map department names to course IDs based on semantic similarity:
#     - CS → 9119 (Informatics Engineering)
#     - Engineering → 9003 (Agronomy) or general engineering
#     - Business → 9147 (Management)
#     - Mathematics → 9003 (Agronomy - requires math) or 9119 (Informatics - uses math)
#     """
#     mapping = {
#         'CS': 9119,              # Informatics Engineering
#         'Engineering': 9003,     # Agronomy (engineering-related)
#         'Business': 9147,        # Management
#         'Mathematics': 9119      # Informatics Engineering (math-heavy)
#     }
#     return mapping.get(department, 9147)  # Default to Management


def map_education_level(education_str):
    """
    Map Parent_Education_Level to graduation dataset's qualification codes:
    - High School → 1 (Secondary Education - 12th Year)
    - Bachelor's → 2 (Higher Education - Bachelor's Degree)
    - Master's → 4 (Higher Education - Master's)
    - PhD → 5 (Higher Education - Doctorate)
    """
    education_str = str(education_str).lower().strip()
    
    if 'phd' in education_str or 'doctorate' in education_str:
        return 5
    elif 'master' in education_str:
        return 4
    elif 'bachelor' in education_str:
        return 2
    elif 'high school' in education_str:
        return 1
    else:
        return 1  # Default to secondary education


def create_semantic_features(df_performance):
    """
    Create semantically mapped features from Students Performance dataset
    to match Student Graduation dataset structure.
    """
    df_mapped = pd.DataFrame()
    
    # 1. Gender: Male=1, Female=0 (matching graduation dataset)
    df_mapped['Gender'] = (df_performance['Gender'] == 'Male').astype(int)
    
    # 2. Age at enrollment
    df_mapped['Age at enrollment'] = df_performance['Age']
    
    # 3. Course (from Department)
    # df_mapped['Course'] = df_performance['Department'].apply(map_department_to_course)
    
    # 4. Mother's and Father's qualification (from Parent_Education_Level)
    # parent_qual = df_performance['Parent_Education_Level'].apply(map_education_level)
    # df_mapped["Mother's qualification"] = parent_qual
    # df_mapped["Father's qualification"] = parent_qual
    
    # 5. Displaced (from Stress_Level: high stress ≈ displaced)
    # Stress >= 8 considered as displaced
    # df_mapped['Displaced'] = (df_performance['Stress_Level (1-10)'] >= 8).astype(int)
    
    # 6. Academic performance mapping
    # Map grades/scores to curricular units (normalized to 0-20 scale)
    df_mapped['Curricular units 1st sem (grade)'] = (df_performance['Quizzes_Avg'] / 100) * 20
    df_mapped['Curricular units 2nd sem (grade)'] = (df_performance['Quizzes_Avg'] / 100) * 20
    
    # Map attendance to approved units (high attendance = more approved)
    # Normalize attendance to 0-6 scale (typical semester units)
    # attendance_norm = df_performance['Attendance (%)'] / 100
    # df_mapped['Curricular units 1st sem (approved)'] = (attendance_norm * 6).round()
    # df_mapped['Curricular units 2nd sem (approved)'] = (attendance_norm * 6).round()
    
    # # Enrolled units (assume 6 per semester for full-time)
    # df_mapped['Curricular units 1st sem (enrolled)'] = 6
    # df_mapped['Curricular units 2nd sem (enrolled)'] = 6
    
    # Evaluations based on participation
    # participation_norm = df_performance['Participation_Score'] / 100
    # df_mapped['Curricular units 1st sem (evaluations)'] = (participation_norm * 6).round()
    # df_mapped['Curricular units 2nd sem (evaluations)'] = (participation_norm * 6).round()
    
    
    
    # 9. Binary features with reasonable defaults
    # df_mapped['Scholarship holder'] = (df_performance['Family_Income_Level'] == 'Low').astype(int)
    df_mapped['Debtor'] = (df_performance['Family_Income_Level'] == 'Low').astype(int)
    df_mapped['Tuition fees up to date'] = (df_performance['Family_Income_Level'] == 'High').astype(int)
    
    
   
    
   
     
    
    # 14. Daytime attendance (assume daytime)
    #df_mapped['Daytime/evening attendance\t'] = (df_performance['Sleep_Hours_per_Night'] >5 ).astype(int)
    
    
    return df_mapped


def prepare_target(df_performance):
    """
    Map Grade to Target (3-class problem):
    - F → 0 (Dropout)
    - E → 1 (Enrolled)
    - A, B, C, D → 2 (Graduate)
    """
    grades = df_performance['Grade'].astype(str).str.upper()
    stress = df_performance['Stress_Level (1-10)']

    # Initialize with NaN to catch unexpected values
    target = pd.Series(np.nan, index=df_performance.index)

    target[grades.isin(['D', 'F'])] = 0  # Dropout
    target[grades.isin(['A', 'B'])] = 2  # Graduate

    c_mask = grades == 'C'
    target[c_mask & (stress > 5)] = 1  # High stress C → Enrolled
    target[c_mask & (stress <= 5)] = 2  # Low stress C → Graduate

    return target.astype(int)


def load_and_prepare_data():
    """Load both datasets and prepare them for transfer learning."""
    print("=" * 70)
    print("TRANSFER LEARNING PIPELINE (SEMANTIC FEATURE MAPPING)")
    print("Students Performance → Student Graduation")
    print("=" * 70)
    
    # Load datasets
    df_performance = pd.read_csv('Students Performance Dataset (1).csv')
    df_train = pd.read_csv('./student-graduation/processed/train.csv')
    df_test = pd.read_csv('./student-graduation/processed/test.csv')
    
    print(f"\n✅ Data loaded successfully")
    print(f"   Performance dataset: {df_performance.shape}")
    print(f"   Graduation train: {df_train.shape}")
    print(f"   Graduation test: {df_test.shape}")
    
    # Prepare targets
    y_source = prepare_target(df_performance).values.astype(int)
    
    # Encode target if it's categorical
    if df_train['Target'].dtype == 'object':
        target_mapping = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
        y_train = df_train['Target'].map(target_mapping).values.astype(int)
        y_test = df_test['Target'].map(target_mapping).values.astype(int)
    else:
        y_train = df_train['Target'].values.astype(int)
        y_test = df_test['Target'].values.astype(int)
    
    # Create semantic features from source
    X_source = create_semantic_features(df_performance)
    
    # Get features from target (drop Target column)
    X_train = df_train.drop(columns=['Target'])
    X_test = df_test.drop(columns=['Target'])
    
    print(f"\n✅ Semantic features created from source dataset")
    print(f"   Source features: {X_source.shape[1]}")
    print(f"   Source feature names (first 10): {list(X_source.columns[:10])}")
    
    # Align features: only use features present in both datasets
    common_features = list(set(X_source.columns) & set(X_train.columns))
    common_features.sort()  # Keep consistent order
    
    print(f"\n✅ Feature alignment:")
    print(f"   Source features: {len(X_source.columns)}")
    print(f"   Target features: {len(X_train.columns)}")
    print(f"   Common features: {len(common_features)}")
    print(f"   Common features: {common_features}")
    
    # Select only common features
    X_source = X_source[common_features]
    X_train = X_train[common_features]
    X_test = X_test[common_features]
    
    print(f"\n✅ Final aligned shapes:")
    print(f"   Source: {X_source.shape[0]} samples, {X_source.shape[1]} features")
    print(f"   Train:  {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"   Test:   {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    return X_source, y_source, X_train, y_train, X_test, y_test


def preprocess_data(X_source, X_train, X_test, y_source):
    """Apply preprocessing: imputation and scaling."""
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_source_imp = imputer.fit_transform(X_source)
    X_train_imp = imputer.transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    # Scale features
    scaler = MinMaxScaler()
    X_source_scaled = scaler.fit_transform(X_source_imp)
    X_train_scaled = scaler.transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    train_processed = X_source.copy()
    train_processed['Target'] = y_source

    train_processed_graduate = train_processed[train_processed['Target'] == 2]
    train_processed_enrolled = train_processed[train_processed['Target'] == 1]
    train_processed_dropout = train_processed[train_processed['Target'] == 0]

    train_processed_enrolled = train_processed_enrolled.sample(random_state=42, n=round(train_processed_dropout.shape[0]), replace=True)
    train_processed_graduate = train_processed_graduate.sample(random_state=42, n=round(train_processed_dropout.shape[0]), replace=True)

    X_source_scaled = pd.concat([train_processed_dropout, train_processed_enrolled, train_processed_graduate]).drop(columns=['Target'])
    y_source = pd.concat([train_processed_dropout, train_processed_enrolled, train_processed_graduate])['Target'].values.astype(int)
    return X_source_scaled, X_train_scaled, X_test_scaled, imputer, scaler, y_source


def pretrain_model(X_source, y_source):
    """Phase 1: Pretrain on Students Performance dataset."""
    print("\n" + "=" * 70)
    print("PHASE 1: PRETRAINING ON STUDENTS PERFORMANCE DATASET")
    print("=" * 70)
    
    n_features = X_source.shape[1]
    print(f"\n📊 Training on {X_source.shape[0]} samples with {n_features} features")
    
    # # Use smaller architecture if few features
    # if n_features < 10:
    #     hidden_layers = (32, 16)
    # elif n_features < 20:
    #     hidden_layers = (64, 32)
    # else:
    #     hidden_layers = (128, 64, 32)
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(64,),
        max_iter=1000,
        random_state=11,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        learning_rate_init=0.01,
        verbose=True,
        activation='tanh',
        alpha=0.01,
        solver='adam'
    )
    
    mlp.fit(X_source, y_source)
    
    train_acc = mlp.score(X_source, y_source)
    print(f"✅ Pretraining complete! Training accuracy: {train_acc:.4f}")
    
    return mlp, X_source, y_source


def finetune_model(pretrained_mlp, X_train, y_train):
    """Phase 2: Finetune on Student Graduation dataset with transferred weights."""
    print("\n" + "=" * 70)
    print("PHASE 2: FINETUNING ON STUDENT GRADUATION DATASET")
    print("=" * 70)
    
    n_features = X_train.shape[1]
    
    # Match architecture to pretrained model
    hidden_layers = pretrained_mlp.hidden_layer_sizes
    
    # Create new model with same architecture
    finetune_mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=15,
        learning_rate_init=0.0001,  # Lower learning rate for finetuning
        warm_start=False,  # Start fresh
        verbose=True
    )
    
    # Fit once to initialize structure with stratified sample
    # from sklearn.model_selection import train_test_split
    # X_init, _, y_init, _ = train_test_split(X_train, y_train, 
    #                                           train_size=min(300, len(X_train) // 2),
    #                                           random_state=42, 
    #                                           stratify=y_train)
    finetune_mlp.fit(X_train, y_train)
    
    print(f"\n📊 Loading pretrained weights and continuing training...")
    
    # Copy weights from pretrained model
    for i in range(len(pretrained_mlp.coefs_)):
        if pretrained_mlp.coefs_[i].shape == finetune_mlp.coefs_[i].shape:
            finetune_mlp.coefs_[i][:] = pretrained_mlp.coefs_[i]
            finetune_mlp.intercepts_[i][:] = pretrained_mlp.intercepts_[i]
    
    # Reset iteration counter and enable warm_start
    finetune_mlp.n_iter_ = 0
    finetune_mlp.t_ = 0
    finetune_mlp.warm_start = True
    
    # Continue training with full dataset
    print(f"📊 Finetuning on {len(X_train)} samples with pretrained weights")
    finetune_mlp.fit(X_train, y_train)
    
    train_acc = finetune_mlp.score(X_train, y_train)
    print(f"✅ Finetuning complete! Training accuracy: {train_acc:.4f}")
    
    return finetune_mlp


def train_baseline(X_train, y_train):
    """Train baseline model without transfer learning."""
    print("\n" + "=" * 70)
    print("BASELINE: TRAINING WITHOUT TRANSFER LEARNING")
    print("=" * 70)
    
    n_features = X_train.shape[1]
    
    if n_features < 10:
        hidden_layers = (32, 16)
    elif n_features < 20:
        hidden_layers = (64, 32)
    else:
        hidden_layers = (128, 64, 32)
    
    baseline_mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        learning_rate_init=0.001,
        verbose=True
    )
    
    print(f"\n📊 Training on {len(X_train)} samples")
    baseline_mlp.fit(X_train, y_train)
    
    train_acc = baseline_mlp.score(X_train, y_train)
    print(f"✅ Baseline training complete! Training accuracy: {train_acc:.4f}")
    
    return baseline_mlp


def evaluate_models(transfer_model, baseline_model, X_test, y_test):
    """Evaluate and compare both models."""
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
    
    print(f"\n📈 IMPROVEMENT:")
    print(f"   Accuracy:          {acc_transfer - acc_baseline:+.4f}")
    print(f"   Balanced Accuracy: {bal_acc_transfer - bal_acc_baseline:+.4f}")
    
    # Classification reports
    target_names = ['Dropout', 'Enrolled', 'Graduate']
    
    print("\n" + "=" * 70)
    print("TRANSFER LEARNING CLASSIFICATION REPORT:")
    print("=" * 70)
    print(classification_report(y_test, y_pred_transfer, target_names=target_names))
    
    print("\n" + "=" * 70)
    print("BASELINE CLASSIFICATION REPORT:")
    print("=" * 70)
    print(classification_report(y_test, y_pred_baseline, target_names=target_names))
    
    # Confusion matrices
    print("\n" + "=" * 70)
    print("CONFUSION MATRICES")
    print("=" * 70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Transfer learning confusion matrix
    cm_transfer = confusion_matrix(y_test, y_pred_transfer)
    disp_transfer = ConfusionMatrixDisplay(confusion_matrix=cm_transfer, display_labels=target_names)
    disp_transfer.plot(ax=axes[0], cmap='Blues', values_format='d')
    axes[0].set_title('Transfer Learning Model\nConfusion Matrix', fontsize=12, fontweight='bold')
    axes[0].grid(False)
    
    # Baseline confusion matrix
    cm_baseline = confusion_matrix(y_test, y_pred_baseline)
    disp_baseline = ConfusionMatrixDisplay(confusion_matrix=cm_baseline, display_labels=target_names)
    disp_baseline.plot(ax=axes[1], cmap='Greens', values_format='d')
    axes[1].set_title('Baseline Model\nConfusion Matrix', fontsize=12, fontweight='bold')
    axes[1].grid(False)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_student_graduation.png', dpi=150, bbox_inches='tight')
    print("\n✅ Confusion matrices saved to: confusion_matrix_student_graduation.png")
    plt.show()


def evaluate_source_performance(model, X_source_scaled, y_source):
    """Evaluate model performance on source dataset (Students Performance)"""
    print("\n" + "=" * 70)
    print("SOURCE DATASET EVALUATION (STUDENTS PERFORMANCE)")
    print("=" * 70)
    
    y_pred_source = model.predict(X_source_scaled)
    
    acc_source = accuracy_score(y_source, y_pred_source)
    bal_acc_source = balanced_accuracy_score(y_source, y_pred_source)
    
    print(f"\n📊 Performance on Students Performance dataset:")
    print(f"   Accuracy:          {acc_source:.4f}")
    print(f"   Balanced Accuracy: {bal_acc_source:.4f}")
    
    # Check which classes are actually present in the source data
    unique_classes = np.unique(y_source)
    target_names = ['Dropout', 'Enrolled', 'Graduate']
    labels = [0, 1, 2]
    
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT - STUDENTS PERFORMANCE:")
    print("=" * 70)
    print(classification_report(y_source, y_pred_source, labels=labels, target_names=target_names, zero_division=0))
    
    # Confusion matrix - use only classes present in data
    cm_source = confusion_matrix(y_source, y_pred_source, labels=labels)
    
    plt.figure(figsize=(8, 6))
    disp_source = ConfusionMatrixDisplay(confusion_matrix=cm_source, display_labels=target_names)
    disp_source.plot(cmap='Purples', values_format='d')
    plt.title('Students Performance Dataset\nConfusion Matrix', fontsize=14, fontweight='bold')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('confusion_matrix_student_performance.png', dpi=150, bbox_inches='tight')
    print("\n✅ Confusion matrix saved to: confusion_matrix_student_performance.png")
    plt.show()


def main():
    """Main execution pipeline."""
    # Step 1: Load and prepare data
    X_source, y_source, X_train, y_train, X_test, y_test = load_and_prepare_data()
    
    if X_source.shape[1] == 0:
        print("\n❌ No common features found! Cannot proceed with transfer learning.")
        print("   Please check feature mapping logic.")
        return
    
    # Step 2: Preprocess data
    X_source_scaled, X_train_scaled, X_test_scaled, imputer, scaler, y_source = preprocess_data(
        X_source, X_train, X_test, y_source
    )
    
    # Step 3: Pretrain model
    pretrained_model, X_source_for_eval, y_source_for_eval = pretrain_model(X_source_scaled, y_source)
    
    # Step 3.5: Evaluate on source dataset
    evaluate_source_performance(pretrained_model, X_source_scaled, y_source)
    
    # Step 4: Finetune model
    transfer_model = finetune_model(pretrained_model, X_train_scaled, y_train)
    
    # Step 5: Train baseline
    baseline_model = train_baseline(X_train_scaled, y_train)
    
    # Step 6: Evaluate on target dataset
    evaluate_models(transfer_model, baseline_model, X_test_scaled, y_test)
    
    print("\n✅ Pipeline completed successfully!")
    print(pd.DataFrame(X_source_scaled).head(30))
    print(pd.DataFrame(y_source).head(30))


if __name__ == "__main__":
    main()
