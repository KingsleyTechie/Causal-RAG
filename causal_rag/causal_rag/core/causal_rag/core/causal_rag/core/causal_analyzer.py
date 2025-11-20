import re
from typing import List, Dict, Tuple
from collections import Counter

class CausalAnalyzer:
    """
    Analyzes retrieved evidence for causal strength and quality.
    """
    
    def __init__(self):
        self.evidence_indicators = {
            'positive': ['yes', 'confirm', 'effective', 'success', 'improve', 'benefit'],
            'negative': ['no', 'not', 'negative', 'ineffective', 'failure', 'worsen', 'harm'],
            'uncertain': ['maybe', 'uncertain', 'inconclusive', 'possibly', 'potentially']
        }
    
    def analyze_evidence_quality(self, contexts: List[str], question: str) -> Dict:
        """
        Analyze the quality and consistency of retrieved evidence.
        
        Args:
            contexts: List of retrieved context passages
            question: Original question
            
        Returns:
            Dictionary with evidence analysis
        """
        analysis = {
            'total_contexts': len(contexts),
            'consistent_yes': 0,
            'consistent_no': 0,
            'weak_evidence': 0,
            'causal_evidence_count': 0,
            'evidence_consistency': 0.0
        }
        
        question_keywords = set(re.findall(r'\w+', question.lower()))
        
        for context in contexts:
            context_lower = context.lower()
            
            # Calculate relevance through keyword overlap
            context_keywords = set(re.findall(r'\w+', context_lower))
            overlap = len(question_keywords.intersection(context_keywords)) / len(question_keywords)
            
            # Check evidence strength
            if overlap > 0.3:  # Good relevance threshold
                if any(word in context_lower for word in self.evidence_indicators['positive']):
                    analysis['consistent_yes'] += 1
                elif any(word in context_lower for word in self.evidence_indicators['negative']):
                    analysis['consistent_no'] += 1
                else:
                    analysis['weak_evidence'] += 1
                
                # Check for causal evidence
                if any(pattern in context_lower for pattern in ['randomized', 'controlled trial', 'RCT']):
                    analysis['causal_evidence_count'] += 1
            else:
                analysis['weak_evidence'] += 1
        
        # Calculate evidence consistency
        total_strong = analysis['consistent_yes'] + analysis['consistent_no']
        if total_strong > 0:
            majority = max(analysis['consistent_yes'], analysis['consistent_no'])
            analysis['evidence_consistency'] = majority / total_strong
        
        return analysis
    
    def determine_answer(self, evidence_analysis: Dict) -> Tuple[str, float]:
        """
        Determine final answer based on evidence analysis.
        
        Args:
            evidence_analysis: Output from analyze_evidence_quality
            
        Returns:
            Tuple of (answer, confidence)
        """
        total_strong = evidence_analysis['consistent_yes'] + evidence_analysis['consistent_no']
        
        # Insufficient evidence
        if total_strong == 0:
            return "maybe", 0.3
        
        # Strong causal evidence available
        if evidence_analysis['causal_evidence_count'] > 0:
            if evidence_analysis['consistent_yes'] > evidence_analysis['consistent_no']:
                confidence = evidence_analysis['evidence_consistency']
                return "yes", max(0.7, confidence)
            elif evidence_analysis['consistent_no'] > evidence_analysis['consistent_yes']:
                confidence = evidence_analysis['evidence_consistency']
                return "no", max(0.7, confidence)
            else:
                return "maybe", 0.5
        
        # Weak evidence case
        if evidence_analysis['consistent_yes'] >= 2:
            return "yes", 0.6
        elif evidence_analysis['consistent_no'] >= 2:
            return "no", 0.6
        else:
            return "maybe", 0.4
