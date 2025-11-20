import pandas as pd
import json
from typing import List, Dict, Any

class DataProcessor:
    """
    Handles data loading and preprocessing for clinical QA tasks.
    """
    
    @staticmethod
    def load_pubmedqa_dataset(file_path: str) -> List[Dict]:
        """
        Load PubMedQA format dataset.
        
        Args:
            file_path: Path to dataset file
            
        Returns:
            List of question-answer pairs
        """
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        else:
            df = pd.read_csv(file_path)
        
        # Convert to standard format
        samples = []
        for _, row in df.iterrows():
            sample = {
                "instruction": row.get('instruction', ''),
                "question": row['input'].replace('Question: ', '').replace('\nAnswer:', ''),
                "answer": row['output']
            }
            samples.append(sample)
        
        return samples
    
    @staticmethod
    def create_synthetic_knowledge_base(questions: List[str]) -> List[str]:
        """
        Create synthetic knowledge base from questions.
        Used for prototyping when real knowledge base is unavailable.
        
        Args:
            questions: List of clinical questions
            
        Returns:
            List of synthetic evidence passages
        """
        knowledge_base = []
        for question in questions:
            evidence = f"Medical research indicates that: {question}. This has been studied in clinical trials and research studies."
            knowledge_base.append(evidence)
        
        return knowledge_base
    
    @staticmethod
    def prepare_knowledge_base(documents: List[str]) -> List[str]:
        """
        Preprocess knowledge base documents.
        
        Args:
            documents: Raw documents
            
        Returns:
            Cleaned and normalized documents
        """
        processed_docs = []
        for doc in documents:
            # Basic cleaning
            doc_clean = doc.strip()
            if doc_clean:
                processed_docs.append(doc_clean)
        
        return processed_docs
