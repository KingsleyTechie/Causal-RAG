# Causal-RAG API Documentation

## Core Classes

### CausalRAGPipeline
The main pipeline class that orchestrates the entire Causal-RAG workflow.

```python
from causal_rag.core import CausalRAGPipeline

# Initialize pipeline
pipeline = CausalRAGPipeline()

# Initialize with knowledge base
knowledge_base = [
    "Medical research indicates that treatment A causes improvement in condition X.",
    "Randomized controlled trial shows drug B reduces symptoms of disease Y."
]
pipeline.initialize(knowledge_base)

# Answer a question
result = pipeline.answer("Does treatment A improve condition X?")
