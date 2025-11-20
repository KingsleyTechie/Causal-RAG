from typing import Dict, List, Optional
from .retriever import CausalRetriever
from .causal_analyzer import CausalAnalyzer
from .generator import CausalGenerator

class CausalRAGPipeline:
    """
    Main pipeline for Causal-RAG framework.
    """
    
    def __init__(self, retriever_model: str = "all-MiniLM-L6-v2"):
        self.retriever = CausalRetriever(model_name=retriever_model)
        self.analyzer = CausalAnalyzer()
        self.generator = CausalGenerator()
        self.is_initialized = False
    
    def initialize(self, knowledge_base: List[str]):
        """
        Initialize the pipeline with a knowledge base.
        
        Args:
            knowledge_base: List of documents to use as knowledge source
        """
        self.retriever.build_index(knowledge_base)
        self.is_initialized = True
    
    def answer(self, question: str, top_k: int = 3) -> Dict:
        """
        Answer a clinical question using causal-enhanced retrieval.
        
        Args:
            question: Clinical question to answer
            top_k: Number of contexts to retrieve
            
        Returns:
            Dictionary containing answer, confidence, and metadata
        """
        if not self.is_initialized:
            raise ValueError("Pipeline not initialized. Call initialize() first.")
        
        # Step 1: Retrieve contexts with causal enhancement
        contexts, retrieval_scores = self.retriever.retrieve(question, top_k=top_k)
        
        # Step 2: Analyze evidence quality
        evidence_analysis = self.analyzer.analyze_evidence_quality(contexts, question)
        
        # Step 3: Generate final answer
        result = self.generator.generate_answer(question, contexts, evidence_analysis)
        
        # Add retrieval metadata
        result.update({
            "retrieval_scores": retrieval_scores,
            "retrieved_count": len(contexts),
            "question": question
        })
        
        return result
    
    def batch_answer(self, questions: List[str], top_k: int = 3) -> List[Dict]:
        """
        Answer multiple questions in batch.
        
        Args:
            questions: List of clinical questions
            top_k: Number of contexts to retrieve per question
            
        Returns:
            List of answer dictionaries
        """
        return [self.answer(q, top_k) for q in questions]
