import unittest
import numpy as np
from src.adaptive_hierarchical_text_clustering.clustering import AdaptiveHierarchicalTextClustering


class TestAdaptiveHierarchicalTextClustering(unittest.TestCase):

    def setUp(self):
        self.clustering = AdaptiveHierarchicalTextClustering(
            threshold_adjustment=0.01,
            window_size=5,
            min_split_tokens=48,
            max_split_tokens=768,
            split_tokens_tolerance=10
        )

    def test_initialization(self):
        self.assertEqual(self.clustering.threshold_adjustment, 0.01)
        self.assertEqual(self.clustering.window_size, 5)
        self.assertEqual(self.clustering.min_split_tokens, 48)
        self.assertEqual(self.clustering.max_split_tokens, 768)
        self.assertEqual(self.clustering.split_tokens_tolerance, 10)

    def test_rolling_similarity_scores(self):
        encoded_docs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 1]])
        expected_scores = [0, 0, 1.0, 0.816497]  # Calculated manually
        scores = self.clustering._rolling_similarity_scores(encoded_docs)
        np.testing.assert_almost_equal(scores, expected_scores, decimal=6)

    def test_find_split_indices(self):
        similarities = [0.9, 0.8, 0.3, 0.7, 0.2, 0.6]
        threshold = 0.5
        expected_splits = [3, 5]
        splits = self.clustering._find_split_indices(similarities, threshold)
        self.assertEqual(splits, expected_splits)

    def test_find_optimal_threshold(self):
        token_counts = [50, 60, 70, 80, 90, 100]
        similarity_scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        threshold = self.clustering._find_optimal_threshold(token_counts, similarity_scores)
        self.assertTrue(0 < threshold < 1)  # Basic sanity check

    def test_fit_simple_case(self):
        X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        T = np.array([50, 60, 70, 80])
        self.clustering.fit(X, T)
        self.assertEqual(len(set(self.clustering.labels_)), 1)  # Should have 1 cluster for this simple case

    def test_tree_structure(self):
        X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        T = np.array([50, 60, 70, 80])
        self.clustering.fit(X, T)
        self.assertIsNotNone(self.clustering.tree_)
        self.assertTrue(hasattr(self.clustering.tree_, 'label'))
        self.assertTrue(hasattr(self.clustering.tree_, 'children'))
