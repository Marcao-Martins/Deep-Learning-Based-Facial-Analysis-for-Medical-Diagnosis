import mlflow
from mlflow.tracking import MlflowClient
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
# os import is not needed if directly using mlflow.log_figure


def set_experiment(experiment_name: str):
    """
    Set or create an MLflow experiment by name.
    """
    mlflow.set_experiment(experiment_name)


def list_experiments() -> List[mlflow.entities.Experiment]:
    """
    Return all MLflow experiments.
    """
    client = MlflowClient()
    return client.list_experiments()


def get_runs(experiment_name: str) -> Any:
    """
    Get all runs for a named experiment.
    """
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    return client.search_runs(experiment_ids=[exp.experiment_id])


def compare_runs(experiment_name: str, metric_key: str, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Return the top N runs sorted by the specified metric in descending order.
    """
    runs = get_runs(experiment_name)
    sorted_runs = sorted(
        runs, key=lambda r: r.data.metrics.get(metric_key, float('-inf')),
        reverse=True
    )
    results: List[Dict[str, Any]] = []
    for r in sorted_runs[:top_n]:
        results.append({
            "run_id": r.info.run_id,
            "metrics": r.data.metrics,
            "params": r.data.params,
            "tags": r.data.tags,
        })
    return results


def log_standard_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    metric_prefix: str,
    num_classes: int,
    y_probs: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Calculates, logs to MLflow, and returns a dictionary of standard classification metrics.

    This function aims to provide a comprehensive set of metrics for classification tasks,
    logging them in a structured way to MLflow for easy comparison across runs and models.

    Args:
        y_true: Numpy array of ground truth labels (integers).
        y_pred: Numpy array of predicted labels (integers).
        class_names: List of class names, ordered by label index.
                     (e.g., class_names[0] corresponds to label 0).
                     If len(class_names) < num_classes, generic names will be used for missing ones.
        metric_prefix: A string prefix for all logged MLflow metrics and artifact names.
                       (e.g., "eval", "test_final", "EfficientNetB0_individual_eval").
        num_classes: The total number of unique classes in the dataset.
                     This is used to ensure the confusion matrix has the correct dimensions
                     and all potential classes are iterated for per-class metrics.
        y_probs: Numpy array of predicted probabilities (optional). Expected shape:
                 - For binary classification: (n_samples, 2) or (n_samples,) [for positive class].
                 - For multi-class: (n_samples, n_classes).
                 Used for ROC AUC calculation.

    Returns:
        A dictionary where keys are metric names (with prefix) and values are the
        calculated metric values. This can be used for further local processing if needed.
    """
    logged_metrics: Dict[str, Any] = {}

    if not isinstance(y_true, np.ndarray): y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray): y_pred = np.array(y_pred)
    if y_probs is not None and not isinstance(y_probs, np.ndarray): y_probs = np.array(y_probs)

    # --- Class Name Handling --- 
    # Ensure class_names array is consistent with num_classes for reliable indexing.
    if len(class_names) < num_classes:
        effective_class_names = [f"Class_{i}" for i in range(num_classes)]
        for i in range(len(class_names)):
            if i < num_classes: # Ensure we don't go out of bounds if class_names is longer but num_classes is smaller
                effective_class_names[i] = class_names[i]
    else:
        # If class_names is longer than num_classes, truncate it.
        effective_class_names = class_names[:num_classes]

    # --- Basic Accuracy --- 
    accuracy = accuracy_score(y_true, y_pred)
    mlflow.log_metric(f"{metric_prefix}_accuracy", accuracy)
    logged_metrics[f"{metric_prefix}_accuracy"] = accuracy

    # --- Labels for sklearn metrics --- 
    # Determine labels present in the data for precision_recall_fscore_support to avoid warnings/errors
    # and ensure metrics are calculated only for relevant classes if some are entirely absent.
    # We use labels from 0 to num_classes-1 for full metrics reporting where possible.
    sklearn_labels_for_avg_metrics = sorted(list(set(y_true) | set(y_pred)))
    # For per-class metrics, we want to iterate through all defined classes.
    all_possible_labels = list(range(num_classes))

    # --- Weighted Precision, Recall, F1-score --- 
    # These are robust to class imbalance.
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', labels=all_possible_labels, zero_division=0
    )
    mlflow.log_metric(f"{metric_prefix}_precision_weighted", precision_w)
    mlflow.log_metric(f"{metric_prefix}_recall_weighted", recall_w)
    mlflow.log_metric(f"{metric_prefix}_f1_weighted", f1_w)
    logged_metrics.update({
        f"{metric_prefix}_precision_weighted": precision_w,
        f"{metric_prefix}_recall_weighted": recall_w,
        f"{metric_prefix}_f1_weighted": f1_w
    })

    # --- Per-class Precision, Recall, F1-score --- 
    precision_pc, recall_pc, f1_pc, support_pc = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=all_possible_labels, zero_division=0
    )
    for i in range(num_classes):
        class_name = effective_class_names[i]
        # precision_pc, recall_pc, f1_pc will have entries for each label in `all_possible_labels`
        mlflow.log_metric(f"{metric_prefix}_{class_name}_precision", precision_pc[i])
        mlflow.log_metric(f"{metric_prefix}_{class_name}_recall", recall_pc[i])
        mlflow.log_metric(f"{metric_prefix}_{class_name}_f1", f1_pc[i])
        if support_pc is not None: # Support might be None if labels arg is not fully representative
             mlflow.log_metric(f"{metric_prefix}_{class_name}_support", support_pc[i])
        logged_metrics.update({
            f"{metric_prefix}_{class_name}_precision": precision_pc[i],
            f"{metric_prefix}_{class_name}_recall": recall_pc[i],
            f"{metric_prefix}_{class_name}_f1": f1_pc[i],
            f"{metric_prefix}_{class_name}_support": support_pc[i] if support_pc is not None else 0
        })

    # --- ROC AUC --- 
    if y_probs is not None:
        # Check if y_true has more than one class for AUC calculation
        if len(np.unique(y_true)) > 1:
            if num_classes == 2:
                # For binary, y_probs can be (n_samples,) [prob of positive class] or (n_samples, 2)
                # If (n_samples, 2), use probabilities of the positive class (typically class 1)
                positive_class_probs = y_probs[:, 1] if y_probs.ndim == 2 and y_probs.shape[1] == 2 else y_probs
                try:
                    roc_auc = roc_auc_score(y_true, positive_class_probs)
                    mlflow.log_metric(f"{metric_prefix}_roc_auc", roc_auc)
                    logged_metrics[f"{metric_prefix}_roc_auc"] = roc_auc
                except ValueError as e:
                    print(f"Warning: Could not compute ROC AUC for binary case ({metric_prefix}): {e}")
                    logged_metrics[f"{metric_prefix}_roc_auc"] = 0.0 # Or float('nan')
            else: # Multi-class ROC AUC (One-vs-Rest)
                try:
                    roc_auc_ovr_weighted = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted', labels=all_possible_labels)
                    mlflow.log_metric(f"{metric_prefix}_roc_auc_ovr_weighted", roc_auc_ovr_weighted)
                    logged_metrics[f"{metric_prefix}_roc_auc_ovr_weighted"] = roc_auc_ovr_weighted
                except ValueError as e:
                    print(f"Warning: Could not compute ROC AUC for multi-class case ({metric_prefix}): {e}")
                    logged_metrics[f"{metric_prefix}_roc_auc_ovr_weighted"] = 0.0 # Or float('nan')
        else:
            print(f"Warning: ROC AUC not computed for {metric_prefix} as y_true contains only one class.")
            logged_metrics[f"{metric_prefix}_roc_auc{'_ovr_weighted' if num_classes > 2 else ''}"] = 0.0

    # --- Confusion Matrix --- 
    # Ensure matrix is num_classes x num_classes for consistency, using all_possible_labels.
    cm = confusion_matrix(y_true, y_pred, labels=all_possible_labels)

    # Plot and log confusion matrix as an image artifact
    fig_cm, ax_cm = plt.subplots(figsize=(max(6, num_classes * 1.2), max(5, num_classes * 1.0)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=effective_class_names,
                yticklabels=effective_class_names,
                ax=ax_cm, cbar=True)
    ax_cm.set_xlabel('Predicted Label')
    ax_cm.set_ylabel('True Label')
    title_prefix_clean = metric_prefix.replace("_", " ").replace("eval ", "").capitalize().strip()
    ax_cm.set_title(f'{title_prefix_clean} Confusion Matrix' if title_prefix_clean else 'Confusion Matrix')
    plt.tight_layout()

    cm_artifact_path = f"evaluation_plots/{metric_prefix}_confusion_matrix.png"
    mlflow.log_figure(fig_cm, cm_artifact_path)
    plt.close(fig_cm) # Important to close figure to free memory
    
    # Log CM cell values as individual metrics for easier querying/comparison in MLflow UI
    for i in range(num_classes):
        for j in range(num_classes):
            true_class_name = effective_class_names[i]
            pred_class_name = effective_class_names[j]
            metric_name = f"{metric_prefix}_cm_{true_class_name}_as_{pred_class_name}"
            value = cm[i, j]
            mlflow.log_metric(metric_name, float(value))
            logged_metrics[metric_name] = float(value)

    # Log Normalized CM cell values (row-wise normalization: P(Predicted=j | True=i) )
    cm_sum_axis1 = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = np.true_divide(cm, cm_sum_axis1)
        cm_norm[~np.isfinite(cm_norm)] = 0
    
    for i in range(num_classes):
        for j in range(num_classes):
            true_class_name = effective_class_names[i]
            pred_class_name = effective_class_names[j]
            metric_name = f"{metric_prefix}_cm_norm_{true_class_name}_as_{pred_class_name}"
            value = cm_norm[i, j]
            mlflow.log_metric(metric_name, float(value))
            logged_metrics[metric_name] = float(value)

    print(f"Standard classification metrics logged with prefix '{metric_prefix}'.")
    return logged_metrics 