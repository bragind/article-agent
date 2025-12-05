import os

def test_data_exists():
    assert os.path.exists("data/articles.jsonl")
