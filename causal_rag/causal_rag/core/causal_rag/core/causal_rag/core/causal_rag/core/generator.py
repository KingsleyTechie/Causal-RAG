from typing import List, Dict, Tuple
import re

class CausalGenerator:
    """
    Generates answers based on causally-enhanced retrieved evidence.
    """
    
    def __init__(self):
        self.answer_templates = {
            "yes": "Based on strong causal evidence from clinical studies, the answer is yes.",
            "no": "Based on the available evidence, the answer is no.",
            "maybe": "The current evidence is insufficient to provide a definitive answer. Further clinical investigation is recommended."
        }
    
    def generate_answer(self, question: str, contexts: List[str], evidence_analysis: Dict) -> Dict:
        """
        Generate final answer with confidence and explanation.
        
        Args:
            question: Original question
            contexts: Retrieved context passages
            evidence_analysis: Analysis from CausalAnalyzer
            
        Returns:
            Dictionary containing answer, confidence, and explanation
        """
        answer, confidence = self._determine_final_answer(evidence_analysis)
        
        explanation = self._generate_explanation(answer, confidence, evidence_analysis, contexts)
        
        return {
            "answer": answer,
            "confidence": confidence,
            "explanation": explanation,
            "evidence_analysis": evidence_analysis,
            "retrieved_contexts": contexts[:2]  # Return top 2 contexts for transparency
        }
    
    def _determine_final_answer(self, evidence_analysis: Dict) -> Tuple[str, float]:
        """Determine final answer based on evidence analysis."""
        total_strong = evidence_analysis['consistent_yes'] + evidence_analysis['consistent_no']
        
        if total_strong == 0:
            return "maybe", 0.3
            
        if evidence_analysis['causal_evidence_count'] >= 1:
            # High confidence for causal evidence
            if evidence_analysis['consistent_yes'] > evidence_analysis['consistent_no']:
                confidence = min(0.9, evidence_analysis['evidence_consistency'] + 0.2)
                return "yes", confidence
            elif evidence_analysis['consistent_no'] > evidence_analysis['consistent_yes']:
                confidence = min(0.9, evidence_analysis['evidence_consistency'] + 0.2)
                return "no", confidence
                
        # Moderate confidence for non-causal evidence
        if evidence_analysis['consistent_yes'] > evidence_analysis['consistent_no']:
            confidence = evidence_analysis['evidence_consistency'] * 0.7
            return "yes", confidence
        elif evidence_analysis['consistent_no'] > evidence_analysis['consistent_yes']:
            confidence = evidence_analysis['evidence_consistency'] * 0.7
            return "no", confidence
            
        return "maybe", 0.4
    
    def _generate_explanation(self, answer: str, confidence: float, 
                           evidence_analysis: Dict, contexts: List[str]) -> str:
        """Generate human-readable explanation for the answer."""
        
        if answer == "yes":
            base = "The evidence supports an affirmative answer."
        elif answer == "no":
            base = "The evidence does not support an affirmative answer."
        else:
            base = "The available evidence is insufficient for a definitive conclusion."
        
        # Add evidence quality information
        if evidence_analysis['causal_evidence_count'] > 0:
            base += f" This conclusion is supported by {evidence_analysis['causal_evidence_count']} source(s) with strong causal evidence."
        else:
            base += " The available sources provide correlational rather than causal evidence."
        
        # Add confidence level
        if confidence > 0.7:
            base += " This is a high-confidence assessment."
        elif confidence > 0.5:
            base += " This is a moderate-confidence assessment."
        else:
            base += " This is a low-confidence assessment and should be verified with additional sources."
        
        return base
