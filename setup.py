from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="adaptive-hierarchical-text-clustering",
    version="0.1.0",
    author="John Ades",
    author_email="john.a.ades@gmail.com",
    description="A library for extracting hierarchical structure from unstructured text using adaptive clustering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/john-ades/adaptive-hierarchical-text-clustering",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "anytree>=2.8.0", "sentence-transformers>=2.0.0"],
    },
)