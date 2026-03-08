from setuptools import setup, find_packages

setup(
    name="chess_tournament",
    version="0.1.0",
    description="Modular chess tournament framework with engines and LLM players",
    packages=find_packages(),
    install_requires=[
        "python-chess",
        "requests==2.32.4",
        "torch",
        "transformers",
        "bitsandbytes",
        "huggingface-hub==1.3.0",
        "smolagents",
        "accelerate",
    ],
    python_requires=">=3.9",
)
