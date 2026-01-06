"""Evaluation metrics for different tasks."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from collections import defaultdict


class ClassificationMetrics:
    """Metrics for text classification tasks."""
    
    @staticmethod
    def compute(
        predictions: np.ndarray,
        labels: np.ndarray,
        average: str = "weighted",
        label_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            predictions: Model predictions
            labels: True labels
            average: Averaging strategy for multi-class
            label_names: Optional label names for detailed report
        
        Returns:
            Dictionary of metrics
        """
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average=average, zero_division=0
        )
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }
        
        # Add per-class metrics if label names provided
        if label_names is not None:
            precision_per_class, recall_per_class, f1_per_class, _ = (
                precision_recall_fscore_support(
                    labels, predictions, average=None, zero_division=0
                )
            )
            
            for i, label_name in enumerate(label_names):
                metrics[f'precision_{label_name}'] = float(precision_per_class[i])
                metrics[f'recall_{label_name}'] = float(recall_per_class[i])
                metrics[f'f1_{label_name}'] = float(f1_per_class[i])
        
        return metrics
    
    @staticmethod
    def get_classification_report(
        predictions: np.ndarray,
        labels: np.ndarray,
        label_names: Optional[List[str]] = None
    ) -> str:
        """Get detailed classification report."""
        return classification_report(
            labels, predictions, target_names=label_names, zero_division=0
        )
    
    @staticmethod
    def get_confusion_matrix(
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """Get confusion matrix."""
        return confusion_matrix(labels, predictions)


class EmbeddingMetrics:
    """Metrics for embedding/retrieval tasks."""
    
    @staticmethod
    def cosine_similarity(
        embeddings1: np.ndarray,
        embeddings2: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings (N x D)
            embeddings2: Second set of embeddings (M x D)
        
        Returns:
            Similarity matrix (N x M)
        """
        # Normalize embeddings
        embeddings1_norm = embeddings1 / (np.linalg.norm(embeddings1, axis=1, keepdims=True) + 1e-8)
        embeddings2_norm = embeddings2 / (np.linalg.norm(embeddings2, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarity
        return np.dot(embeddings1_norm, embeddings2_norm.T)
    
    @staticmethod
    def mean_reciprocal_rank(
        similarities: np.ndarray,
        relevant_indices: np.ndarray
    ) -> float:
        """
        Compute Mean Reciprocal Rank.
        
        Args:
            similarities: Similarity scores (N x M)
            relevant_indices: Indices of relevant items for each query
        
        Returns:
            MRR score
        """
        reciprocal_ranks = []
        
        for i, sim_row in enumerate(similarities):
            # Sort indices by similarity (descending)
            sorted_indices = np.argsort(-sim_row)
            
            # Find rank of relevant document
            relevant_idx = relevant_indices[i]
            rank = np.where(sorted_indices == relevant_idx)[0][0] + 1
            reciprocal_ranks.append(1.0 / rank)
        
        return float(np.mean(reciprocal_ranks))
    
    @staticmethod
    def recall_at_k(
        similarities: np.ndarray,
        relevant_indices: np.ndarray,
        k: int = 10
    ) -> float:
        """
        Compute Recall@K.
        
        Args:
            similarities: Similarity scores (N x M)
            relevant_indices: Indices of relevant items for each query
            k: Number of top items to consider
        
        Returns:
            Recall@K score
        """
        hits = 0
        
        for i, sim_row in enumerate(similarities):
            # Get top-k indices
            top_k_indices = np.argsort(-sim_row)[:k]
            
            # Check if relevant document is in top-k
            if relevant_indices[i] in top_k_indices:
                hits += 1
        
        return float(hits / len(similarities))
    
    @staticmethod
    def ndcg_at_k(
        similarities: np.ndarray,
        relevant_indices: np.ndarray,
        k: int = 10
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain at K.
        
        Args:
            similarities: Similarity scores (N x M)
            relevant_indices: Indices of relevant items for each query
            k: Number of top items to consider
        
        Returns:
            NDCG@K score
        """
        ndcg_scores = []
        
        for i, sim_row in enumerate(similarities):
            # Get top-k indices
            sorted_indices = np.argsort(-sim_row)[:k]
            
            # Create relevance vector (1 for relevant, 0 otherwise)
            relevance = np.zeros(k)
            relevant_idx = relevant_indices[i]
            
            if relevant_idx in sorted_indices:
                pos = np.where(sorted_indices == relevant_idx)[0][0]
                relevance[pos] = 1
            
            # Compute DCG
            dcg = np.sum(relevance / np.log2(np.arange(2, k + 2)))
            
            # Compute IDCG (best case: relevant at position 0)
            idcg = 1.0 / np.log2(2)
            
            # Compute NDCG
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)
        
        return float(np.mean(ndcg_scores))
    
    @staticmethod
    def compute(
        query_embeddings: np.ndarray,
        doc_embeddings: np.ndarray,
        relevant_indices: np.ndarray,
        k_values: List[int] = [1, 5, 10, 20]
    ) -> Dict[str, float]:
        """
        Compute all embedding metrics.
        
        Args:
            query_embeddings: Query embeddings (N x D)
            doc_embeddings: Document embeddings (M x D)
            relevant_indices: Indices of relevant documents for each query
            k_values: List of k values for Recall@K and NDCG@K
        
        Returns:
            Dictionary of metrics
        """
        # Compute similarities
        similarities = EmbeddingMetrics.cosine_similarity(
            query_embeddings, doc_embeddings
        )
        
        # Compute metrics
        metrics = {
            'mrr': EmbeddingMetrics.mean_reciprocal_rank(similarities, relevant_indices),
        }
        
        for k in k_values:
            metrics[f'recall@{k}'] = EmbeddingMetrics.recall_at_k(
                similarities, relevant_indices, k
            )
            metrics[f'ndcg@{k}'] = EmbeddingMetrics.ndcg_at_k(
                similarities, relevant_indices, k
            )
        
        return metrics


class NERMetrics:
    """Metrics for Named Entity Recognition tasks."""
    
    @staticmethod
    def compute_token_level(
        predictions: List[List[str]],
        labels: List[List[str]]
    ) -> Dict[str, float]:
        """
        Compute token-level metrics.
        
        Args:
            predictions: Predicted tags for each sequence
            labels: True tags for each sequence
        
        Returns:
            Dictionary of metrics
        """
        # Flatten predictions and labels
        flat_predictions = [tag for seq in predictions for tag in seq]
        flat_labels = [tag for seq in labels for tag in seq]
        
        # Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            flat_labels, flat_predictions, average='weighted', zero_division=0
        )
        
        accuracy = accuracy_score(flat_labels, flat_predictions)
        
        return {
            'token_accuracy': float(accuracy),
            'token_precision': float(precision),
            'token_recall': float(recall),
            'token_f1': float(f1),
        }
    
    @staticmethod
    def extract_entities(
        tokens: List[str],
        tags: List[str]
    ) -> List[Tuple[str, int, int, str]]:
        """
        Extract entities from BIO/BIOES tagged sequence.
        
        Args:
            tokens: List of tokens
            tags: List of BIO/BIOES tags
        
        Returns:
            List of entities (entity_type, start_idx, end_idx, entity_text)
        """
        entities = []
        current_entity = None
        
        for i, (token, tag) in enumerate(zip(tokens, tags)):
            if tag.startswith('B-'):
                # Start of new entity
                if current_entity is not None:
                    entities.append(current_entity)
                entity_type = tag[2:]
                current_entity = [entity_type, i, i + 1, token]
            
            elif tag.startswith('I-'):
                # Inside entity
                if current_entity is not None:
                    current_entity[2] = i + 1
                    current_entity[3] += ' ' + token
            
            else:
                # Outside entity (O tag)
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
        
        # Add last entity if exists
        if current_entity is not None:
            entities.append(current_entity)
        
        return [(e[0], e[1], e[2], e[3]) for e in entities]
    
    @staticmethod
    def compute_entity_level(
        predictions: List[List[str]],
        labels: List[List[str]],
        tokens: List[List[str]]
    ) -> Dict[str, float]:
        """
        Compute entity-level metrics.
        
        Args:
            predictions: Predicted tags for each sequence
            labels: True tags for each sequence
            tokens: Tokens for each sequence
        
        Returns:
            Dictionary of metrics
        """
        total_pred = 0
        total_true = 0
        total_correct = 0
        
        entity_type_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        for pred_tags, true_tags, toks in zip(predictions, labels, tokens):
            # Extract entities
            pred_entities = set(NERMetrics.extract_entities(toks, pred_tags))
            true_entities = set(NERMetrics.extract_entities(toks, true_tags))
            
            total_pred += len(pred_entities)
            total_true += len(true_entities)
            total_correct += len(pred_entities & true_entities)
            
            # Per entity type stats
            for entity in pred_entities:
                entity_type = entity[0]
                if entity in true_entities:
                    entity_type_stats[entity_type]['tp'] += 1
                else:
                    entity_type_stats[entity_type]['fp'] += 1
            
            for entity in true_entities:
                entity_type = entity[0]
                if entity not in pred_entities:
                    entity_type_stats[entity_type]['fn'] += 1
        
        # Compute overall metrics
        precision = total_correct / total_pred if total_pred > 0 else 0
        recall = total_correct / total_true if total_true > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'entity_precision': float(precision),
            'entity_recall': float(recall),
            'entity_f1': float(f1),
        }
        
        # Add per-type metrics
        for entity_type, stats in entity_type_stats.items():
            tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
            type_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            type_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            type_f1 = 2 * type_precision * type_recall / (type_precision + type_recall) if (type_precision + type_recall) > 0 else 0
            
            metrics[f'{entity_type}_precision'] = float(type_precision)
            metrics[f'{entity_type}_recall'] = float(type_recall)
            metrics[f'{entity_type}_f1'] = float(type_f1)
        
        return metrics
    
    @staticmethod
    def compute(
        predictions: List[List[str]],
        labels: List[List[str]],
        tokens: List[List[str]]
    ) -> Dict[str, float]:
        """
        Compute all NER metrics.
        
        Args:
            predictions: Predicted tags for each sequence
            labels: True tags for each sequence
            tokens: Tokens for each sequence
        
        Returns:
            Dictionary of metrics
        """
        token_metrics = NERMetrics.compute_token_level(predictions, labels)
        entity_metrics = NERMetrics.compute_entity_level(predictions, labels, tokens)
        
        return {**token_metrics, **entity_metrics}
