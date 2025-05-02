import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import precision_score, recall_score, ndcg_score

class RecommenderEvaluator:
    @staticmethod
    def calculate_metrics(model, test_interactions, test_weights, k: int = 10) -> Dict[str, float]:
        """Calculate precision@k, recall@k, and NDCG@k"""
        
        # Get all user indices from test set
        test_users = np.unique(test_interactions.row)
        
        # Initialize metrics
        precision_at_k = []
        recall_at_k = []
        ndcg_at_k = []
        
        for user_idx in test_users:
            # Get actual items for this user
            actual_items = test_interactions.indices[test_interactions.row == user_idx]
            
            if len(actual_items) == 0:
                continue
                
            # Get predicted scores for all items
            scores = model.predict(user_idx, np.arange(test_interactions.shape[1]))
            
            # Get top k items
            top_items = np.argsort(-scores)[:k]
            
            # Calculate metrics
            precision = len(set(top_items) & set(actual_items)) / k
            recall = len(set(top_items) & set(actual_items)) / len(actual_items)
            
            # Calculate NDCG
            relevance = np.in1d(top_items, actual_items).astype(float)
            ndcg = ndcg_score([relevance], [np.ones_like(relevance)])
            
            precision_at_k.append(precision)
            recall_at_k.append(recall)
            ndcg_at_k.append(ndcg)
        
        return {
            f'precision@{k}': np.mean(precision_at_k),
            f'recall@{k}': np.mean(recall_at_k),
            f'ndcg@{k}': np.mean(ndcg_at_k)
        } 