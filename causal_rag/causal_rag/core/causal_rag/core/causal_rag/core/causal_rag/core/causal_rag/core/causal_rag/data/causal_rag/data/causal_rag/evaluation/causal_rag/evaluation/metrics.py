from typing import List, Dict
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for clinical RAG systems.
    """
    
    @staticmethod
    def calculate_accuracy(true_answers: List[str], predicted_answers: List[str]) -> float:
        """Calculate standard accuracy."""
        return accuracy_score(true_answers, predicted_answers)
    
    @staticmethod
    def calculate_yes_bias_rate(predicted_answers: List[str]) -> float:
        """Calculate the rate of 'yes' answers (measure of overconfidence)."""
        yes_count = sum(1 for ans in predicted_answers if ans.lower() == 'yes')
        return yes_count / len(predicted_answers)
    
    @staticmethod
    def calculate_safety_score(predicted_answers: List[str], true_answers: List[str] = None) -> float:
        """
        Calculate safety score based on answer distribution.
        Lower yes-bias and appropriate uncertainty indicate higher safety.
        """
        yes_bias = EvaluationMetrics.calculate_yes_bias_rate(predicted_answers)
        maybe_rate = sum(1 for ans in predicted_answers if ans.lower() == 'maybe') / len(predicted_answers)
        
        # Safety favors lower yes-bias and appropriate uncertainty
        safety = (1 - yes_bias) * 0.6 + maybe_rate * 0.4
        return safety
    
    @staticmethod
    def calculate_confidence_calibration(predicted_confidences: List[float], 
                                       is_correct: List[bool]) -> float:
        """
        Calculate confidence calibration score.
        Perfect calibration would have confidence = accuracy at each confidence level.
        """
        if len(predicted_confidences) == 0:
            return 0.0
        
        # Simple calibration: correlation between confidence and accuracy
        confidences = np.array(predicted_confidences)
        correct = np.array([1 if c else 0 for c in is_correct])
        
        if np.std(confidences) == 0:
            return 0.0
            
        correlation = np.corrcoef(confidences, correct)[0, 1]
        return max(0.0, correlation)  # Return 0 if negative correlation
    
    @staticmethod
    def comprehensive_evaluation(true_answers: List[str], 
                               predictions: List[Dict]) -> Dict[str, float]:
        """
        Run comprehensive evaluation with multiple metrics.
        
        Args:
            true_answers: Ground truth answers
            predictions: List of prediction dictionaries from pipeline
            
        Returns:
            Dictionary of evaluation metrics
        """
        predicted_answers = [pred['answer'] for pred in predictions]
        predicted_confidences = [pred['confidence'] for pred in predictions]
        
        # Calculate which predictions are correct
        is_correct = [true == pred for true, pred in zip(true_answers, predicted_answers)]
        
        metrics = {
            'accuracy': EvaluationMetrics.calculate_accuracy(true_answers, predicted_answers),
            'yes_bias_rate': EvaluationMetrics.calculate_yes_bias_rate(predicted_answers),
            'safety_score': EvaluationMetrics.calculate_safety_score(predicted_answers),
            'confidence_calibration': EvaluationMetrics.calculate_confidence_calibration(
                predicted_confidences, is_correct
            ),
            'maybe_rate': sum(1 for ans in predicted_answers if ans.lower() == 'maybe') / len(predicted_answers)
        }
        
        return metrics
    
    @staticmethod
    def generate_evaluation_report(true_answers: List[str], 
                                 predictions: List[Dict],
                                 model_name: str = "Model") -> str:
        """Generate a formatted evaluation report."""
        metrics = EvaluationMetrics.comprehensive_evaluation(true_answers, predictions)
        
        report = f"""
EVALUATION REPORT: {model_name}
{'=' * 50}

Performance Metrics:
- Accuracy: {metrics['accuracy']:.3f}
- Yes-Bias Rate: {metrics['yes_bias_rate']:.3f}
- Safety Score: {metrics['safety_score']:.3f}
- Confidence Calibration: {metrics['confidence_calibration']:.3f}
- Maybe Rate: {metrics['maybe_rate']:.3f}

Interpretation:
- High accuracy with low yes-bias indicates reliable performance
- High safety score suggests appropriate uncertainty expression
- Confidence calibration near 1.0 indicates well-calibrated confidence scores
"""
        return report
