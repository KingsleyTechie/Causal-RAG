import pytest
from causal_rag.core.causal_analyzer import CausalAnalyzer

class TestCausalAnalyzer:
    """Test cases for CausalAnalyzer."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = CausalAnalyzer()
        assert 'positive' in analyzer.evidence_indicators
        assert 'negative' in analyzer.evidence_indicators
    
    def test_evidence_analysis(self):
        """Test evidence quality analysis."""
        analyzer = CausalAnalyzer()
        contexts = [
            "Yes, the treatment is effective according to randomized trial.",
            "Study confirms positive outcomes.",
            "Research shows no significant benefit."
        ]
        
        analysis = analyzer.analyze_evidence_quality(contexts, "Is treatment effective?")
        
        assert analysis['total_contexts'] == 3
        assert analysis['consistent_yes'] > 0
        assert analysis['consistent_no'] > 0
    
    def test_answer_determination(self):
        """Test answer determination logic."""
        analyzer = CausalAnalyzer()
        
        # Test strong evidence case
        strong_analysis = {
            'consistent_yes': 3,
            'consistent_no': 0,
            'causal_evidence_count': 2,
            'evidence_consistency': 1.0
        }
        
        answer, confidence = analyzer.determine_answer(strong_analysis)
        assert answer in ['yes', 'no', 'maybe']
        assert 0 <= confidence <= 1
        
        # Test weak evidence case
        weak_analysis = {
            'consistent_yes': 1, 
            'consistent_no': 1,
            'causal_evidence_count': 0,
            'evidence_consistency': 0.5
        }
        
        answer, confidence = analyzer.determine_answer(weak_analysis)
        assert answer == 'maybe'
        assert confidence < 0.7
