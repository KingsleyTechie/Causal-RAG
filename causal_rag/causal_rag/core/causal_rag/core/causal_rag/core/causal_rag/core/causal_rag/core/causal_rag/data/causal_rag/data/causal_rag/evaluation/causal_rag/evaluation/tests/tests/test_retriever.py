import pytest
import numpy as np
from causal_rag.core.retriever import CausalRetriever

class TestCausalRetriever:
    """Test cases for CausalRetriever."""
    
    def test_initialization(self):
        """Test retriever initialization."""
        retriever = CausalRetriever()
        assert retriever.encoder is not None
        assert retriever.index is None
        assert len(retriever.causal_patterns) > 0
    
    def test_build_index(self):
        """Test index building."""
        retriever = CausalRetriever()
        documents = [
            "Medical research indicates that treatment A causes improvement.",
            "Study shows correlation between factor B and outcome C.",
            "Randomized controlled trial demonstrates efficacy."
        ]
        
        retriever.build_index(documents)
        assert retriever.index is not None
        assert retriever.index.ntotal == 3
    
    def test_causal_scoring(self):
        """Test causal evidence scoring."""
        retriever = CausalRetriever()
        
        # Test strong causal text
        strong_text = "Randomized controlled trial shows treatment causes improvement."
        strong_score = retriever._calculate_causal_score(strong_text)
        assert strong_score >= 3
        
        # Test weak causal text  
        weak_text = "Study observes correlation between factors."
        weak_score = retriever._calculate_causal_score(weak_text)
        assert weak_score < strong_score
    
    def test_retrieval(self):
        """Test retrieval functionality."""
        retriever = CausalRetriever()
        documents = [
            "Heart medication reduces risk of myocardial infarction.",
            "Study shows correlation between diet and heart health.", 
            "Randomized trial proves drug efficacy for cardiac patients."
        ]
        
        retriever.build_index(documents)
        results, scores = retriever.retrieve("heart treatment", top_k=2)
        
        assert len(results) == 2
        assert len(scores) == 2
        assert all(isinstance(score, float) for score in scores)
