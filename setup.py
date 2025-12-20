from setuptools import setup, find_packages

setup(
    name="rag-agent-dreamteamru",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.29.0",
        "pandas>=2.1.4",
        "plotly>=5.18.0",
        "pdfplumber>=0.10.3",
        "aiogram>=3.2.0",
        "python-dotenv>=1.0.0",
        "sentence-transformers>=2.2.2",
        "beautifulsoup4>=4.12.2",
        "requests>=2.31.0",
        "chromadb>=0.4.22",
        "numpy>=1.24.3",
        "pydantic>=2.5.0",
        "loguru>=0.7.2",
    ],
    python_requires=">=3.8",
)