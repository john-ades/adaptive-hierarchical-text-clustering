# AdaptiveHierarchicalTextClustering

AdaptiveHierarchicalTextClustering is a Python library for extracting hierarchical structure from unstructured text using an adaptive clustering approach. This project aims to provide an efficient and flexible way to organize and understand large volumes of text data by creating meaningful hierarchies.

## Features

- Adaptive threshold selection for optimal clustering
- Rolling window approach for context-aware similarity calculation
- Token-aware splitting to maintain coherent text segments
- Hierarchical clustering with tree structure output
- Easy integration with popular NLP libraries like sentence-transformers

## Installation

To install AdaptiveHierarchicalTextClustering, run the following command:

```bash
pip install adaptive-hierarchical-text-clustering
```

## Quick Start

Here's a simple example of how to use AdaptiveHierarchicalTextClustering:

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from adaptive_hierarchical_text_clustering import AdaptiveHierarchicalTextClustering

# Prepare your text data
sentences = ["Your", "list", "of", "sentences", "here"]

# Encode sentences (using sentence-transformers as an example)
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(sentences)

# Calculate token counts (simple approximation)
token_counts = [len(sentence.split()) for sentence in sentences]

# Initialize and fit the clustering model
clustering = AdaptiveHierarchicalTextClustering(
    threshold_adjustment=0.01,
    window_size=3,
    min_split_tokens=10,
    max_split_tokens=50,
    split_tokens_tolerance=5
)
clustering.fit(embeddings, np.array(token_counts))

# Access clustering results
print(clustering.labels_)
print(clustering.tree_)
```

For a more detailed example, including visualization of the hierarchical structure, see the `examples/hierarchy_clustering.py` file.

## How It Works

AdaptiveHierarchicalTextClustering works by:

1. Calculating similarity scores between text segments using a rolling window approach
2. Adaptively finding an optimal threshold for clustering based on token counts
3. Building a hierarchical structure of clusters
4. Providing both flat cluster labels and a tree structure of the hierarchy

The algorithm is particularly suited for tasks where maintaining context and creating a meaningful hierarchy are important.

## Use Cases

- Document summarization: Create hierarchical summaries of long documents
- Topic modeling: Discover hierarchical topic structures in large text corpora
- Content organization: Automatically organize and structure content for websites or knowledge bases
- Text segmentation: Identify coherent segments within long texts
- Hierarchical text classification: Create multi-level classification systems for text data

## Contributing

Contributions are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was inspired by the need for better hierarchical text organization in various NLP tasks.
- Thanks to the semantic-router for providing inspiration for the essential functionalities used in this project.

## Contact

If you have any questions or feedback, please open an issue on the GitHub repository.