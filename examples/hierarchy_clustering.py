import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except:
    raise RuntimeError("For this example we use the 'anytree' library. Run 'pip install sentence_transformers' to continue")
try:
    from anytree import Node, RenderTree
except:
    raise RuntimeError("For this example we use the 'anytree' library. Run 'pip install anytree' to continue")

from src.adaptive_hierarchical_text_clustering.clustering import AdaptiveHierarchicalTextClustering

def split_sentences(text):
    """Simple sentence splitting function."""
    return [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]

def create_anytree(cluster_tree, sentences):
    """Convert cluster tree to anytree structure."""
    def build_tree(node, parent=None):
        tree_node = Node(f"Cluster {node.label}", parent=parent)
        if not node.children:
            tree_node.name = sentences[node.label]
        for child in node.children:
            build_tree(child, tree_node)
        return tree_node
    return build_tree(cluster_tree)

# Example text (a short excerpt about AI)
text = """
Artificial intelligence is reshaping the world as we know it. Machine learning algorithms are becoming increasingly sophisticated, 
capable of analyzing vast amounts of data and making predictions with remarkable accuracy. Natural language processing has made 
significant strides, enabling machines to understand and generate human-like text. Computer vision systems can now recognize and 
interpret visual information with precision rivaling human capabilities. These advancements are driving innovation across industries, 
from healthcare and finance to transportation and entertainment. As AI continues to evolve, it raises important questions about ethics, 
privacy, and the future of work. Researchers and policymakers are grappling with the challenge of ensuring that AI development 
benefits humanity as a whole. The potential of AI is immense, but so too are the responsibilities that come with its development and deployment.
"""

if __name__ == '__main__':

    # Split the text into sentences
    sentences = split_sentences(text)

    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode sentences
    embeddings = model.encode(sentences)

    # Calculate token counts (using a simple approximation)
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

    # Create anytree structure
    root = create_anytree(clustering.tree_, sentences)

    # Print the tree
    for pre, _, node in RenderTree(root):
        print(f"{pre}{node.name}")
    # output:
    # Cluster 0
    # ├── Cluster 0
    # │   └── Cluster 0
    # │       └── Cluster 0
    # │           ├── Artificial intelligence is reshaping the world as we know it
    # │           └── Machine learning algorithms are becoming increasingly sophisticated,  capable of analyzing vast amounts of data and making predictions with remarkable accuracy
    # ├── Cluster 2
    # │   └── Cluster 2
    # │       └── Cluster 2
    # │           ├── Natural language processing has made  significant strides, enabling machines to understand and generate human-like text
    # │           └── Computer vision systems can now recognize and  interpret visual information with precision rivaling human capabilities
    # └── Cluster 4
    #     └── Cluster 4
    #         ├── Cluster 4
    #         │   └── These advancements are driving innovation across industries,  from healthcare and finance to transportation and entertainment
    #         └── Cluster 5
    #             ├── As AI continues to evolve, it raises important questions about ethics,  privacy, and the future of work
    #             ├── Researchers and policymakers are grappling with the challenge of ensuring that AI development  benefits humanity as a whole
    #             └── The potential of AI is immense, but so too are the responsibilities that come with its development and deployment
