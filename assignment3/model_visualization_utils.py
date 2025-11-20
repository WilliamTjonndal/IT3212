import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def plot_feature_importance_comparison(models_dict, feature_names, top_n=15):
    """
    Plot feature importances side-by-side for tree-based models.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model names as keys and fitted models as values
    feature_names : list
        List of feature names corresponding to the training data
    top_n : int
        Number of top features to display
    """
    tree_models = {}
    
    # Filter only tree-based models with feature_importances_
    for name, model in models_dict.items():
        if hasattr(model.named_steps['model'], 'feature_importances_'):
            tree_models[name] = model.named_steps['model'].feature_importances_
    
    if not tree_models:
        print("No tree-based models with feature importances found.")
        return
    
    n_models = len(tree_models)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, importances) in zip(axes, tree_models.items()):
        # Get top N features
        indices = np.argsort(importances)[-top_n:]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        # Create horizontal bar plot
        ax.barh(range(len(top_features)), top_importances, color='steelblue')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Features - {name}')
        ax.grid(axis='x', alpha=0.3)
        
        # Add values on bars
        for i, v in enumerate(top_importances):
            ax.text(v, i, f' {v:.4f}', va='center')
    
    plt.tight_layout()
    plt.show()


def plot_class_performance_heatmap(models_dict, X_test, y_test):
    """
    Create a heatmap showing per-class F1-scores for all models.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model names as keys and fitted models as values
    X_test : DataFrame
        Test features
    y_test : Series
        Test labels
    """
    from sklearn.metrics import f1_score
    import pandas as pd
    import seaborn as sns
    
    classes = np.unique(y_test)
    results = []
    
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        f1_scores = f1_score(y_test, y_pred, average=None)
        results.append(f1_scores)
    
    # Create DataFrame
    df_results = pd.DataFrame(
        results,
        index=list(models_dict.keys()),
        columns=[f'Class {c}' for c in classes]
    )
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_results, annot=True, fmt='.3f', cmap='YlGnBu', 
                cbar_kws={'label': 'F1-Score'})
    plt.title('Per-Class F1-Score Heatmap')
    plt.ylabel('Model')
    plt.xlabel('Class')
    plt.tight_layout()
    plt.show()
    
    return df_results


def plot_confusion_matrices_grid(models_dict, X_test, y_test):
    """
    Plot confusion matrices for all models in a 2x2 grid.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model names as keys and fitted models as values
    X_test : DataFrame
        Test features
    y_test : Series
        Test labels
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    n_models = len(models_dict)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for ax, (name, model) in zip(axes, models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(name)
    
    # Hide unused subplots if fewer than 4 models
    for i in range(n_models, 4):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def create_model_comparison_table(models_dict, X_test, y_test):
    """
    Create a comprehensive comparison table of all models.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model names as keys and fitted models as values
    X_test : DataFrame
        Test features
    y_test : Series
        Test labels
    
    Returns:
    --------
    DataFrame with comparison metrics
    """
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, 
                                  precision_score, recall_score, f1_score,
                                  roc_auc_score)
    import pandas as pd
    
    results = []
    
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        
        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred),
            'Precision (macro)': precision_score(y_test, y_pred, average='macro'),
            'Recall (macro)': recall_score(y_test, y_pred, average='macro'),
            'F1-Score (macro)': f1_score(y_test, y_pred, average='macro')
        }
        
        # Add ROC-AUC if model supports predict_proba
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            metrics['ROC-AUC (OvR)'] = roc_auc_score(y_test, y_proba, multi_class='ovr')
        else:
            metrics['ROC-AUC (OvR)'] = np.nan
        
        results.append(metrics)
    
    df_comparison = pd.DataFrame(results)
    df_comparison = df_comparison.round(4)
    
    return df_comparison