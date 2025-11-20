import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
import re

class CausalRetriever:
    """
    A retriever that enhances semantic search with causal evidence prioritization.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.knowledge_base = []
        
        # Causal language patterns for evidence scoring
        self.causal_patterns = {
            'strong_causal': [
                r'randomized.*controlled', r'RCT', r'clinical trial', r'double-blind',
                r'placebo-controlled', r'intervention', r'treatment effect',
                r'causes?', r'causal', r'mechanism', r'pathway'
            ],
            'moderate_causal': [
                r'associated with', r'correlated with', r'related to', r'predicts?',
                r'indicat', r'marker', r'risk factor', r'effect of', r'impact of'
            ]
        }
    
    def build_index(self, documents: List[str]):
        """Build FAISS index from documents."""
        self.knowledge_base = documents
        embeddings = self.encoder.encode(documents, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
    
    def _calculate_causal_score(self, text: str) -> float:
        """Calculate causal evidence score for a text passage."""
        score = 0
        text_lower = text.lower()
        
        for pattern in self.causal_patterns['strong_causal']:
            if re.search(pattern, text_lower):
                score += 3
                break
                
        for pattern in self.causal_patterns['moderate_causal']:
            if re.search(pattern, text_lower):
                score += 2
                break
        
        return score
    
    def retrieve(self, query: str, top_k: int = 3, causal_weight: float = 0.5) -> Tuple[List[str], List[float]]:
        """
        Retrieve documents with causal enhancement.
        
        Args:
            query: Input query
            top_k: Number of documents to retrieve
            causal_weight: Weight for causal scoring vs semantic similarity
            
        Returns:
            Tuple of (retrieved_documents, combined_scores)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Encode query
        query_embedding = self.encoder.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Get initial candidates (more than needed)
        initial_k = top_k * 3
        semantic_scores, indices = self.index.search(query_embedding, initial_k)
        semantic_scores = semantic_scores[0]
        indices = indices[0]
        
        # Score candidates with causal enhancement
        scored_documents = []
        for idx, semantic_score in zip(indices, semantic_scores):
            document = self.knowledge_base[idx]
            causal_score = self._calculate_causal_score(document)
            combined_score = semantic_score + (causal_score * causal_weight)
            scored_documents.append((document, combined_score, causal_score))
        
        # Sort by combined score and return top_k
        scored_documents.sort(key=lambda x: x[1], reverse=True)
        final_documents = [doc[0] for doc in scored_documents[:top_k]]
        final_scores = [doc[1] for doc in scored_documents[:top_k]]
        
        return final_documents, final_scores
