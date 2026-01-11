import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import EvalPrediction

def compute_metrics(eval_pred: EvalPrediction):
    """
    Compute metrics for both standard and SCE tasks.
    eval_pred.predictions: logits [B, K] or [B, NumLabels]
    eval_pred.label_ids: labels [B] (Target Index for SCE or Label ID)
    """
    logits, labels = eval_pred
    
    # Handle tuple output (loss, logits) if Trainer returns it
    if isinstance(logits, tuple):
        logits = logits[0]
        
    predictions = np.argmax(logits, axis=-1)
    
    # Standard Accuracy
    acc = accuracy_score(labels, predictions)
    
    # F1 Score (Macro/Micro)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    f1_micro = f1_score(labels, predictions, average="micro", zero_division=0)
    
    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro
    }
    
    # For SCE / Top-k support (if meaningful)
    # If logits shape is large (K candidates), we can compute top-k
    if logits.shape[1] > 2:
        # Top-3 Accuracy
        k = min(3, logits.shape[1])
        # argsort gives ascending order, take last k
        top_k_preds = np.argsort(logits, axis=-1)[:, -k:]
        # check if label in top_k
        top_k_acc = np.mean([1 if l in p else 0 for l, p in zip(labels, top_k_preds)])
        metrics[f"top_{k}_accuracy"] = top_k_acc
        
    return metrics
