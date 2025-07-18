"""Setup script for pdf-chef package."""

from setuptools import find_packages, setup

setup(
    name="pdf-chef",
    version="0.1.0",
    description="LLM chatbot that interfaces with PDFs for technical documentation",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "click>=8.1.3",
        "langchain>=0.0.267",
        "langchain-openai>=0.0.2",
        "langchain-community>=0.0.1",
        "openai>=1.6.0",
        "pymupdf>=1.22.5",
        "python-dotenv>=1.0.0",
        "rich>=13.4.2",
        "chromadb>=0.4.13",
        "tqdm>=4.65.0",
        "tiktoken>=0.4.0",
    ],
    entry_points={
        "console_scripts": [
            "pdf-chef=pdf_chef.cli:main",
        ],
    },
    python_requires=">=3.12",
)
