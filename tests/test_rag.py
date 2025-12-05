import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag import RAGAgent

def test_rag_initialization():
    agent = RAGAgent()
    assert agent is not None
