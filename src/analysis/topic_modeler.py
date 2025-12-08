"""
Topic Modeler using BERTopic for automatic aspect discovery.

Uses BERTopic to extract topics from reviews, which serve as
automatically discovered aspects for analysis.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TopicModeler:
    """
    BERTopic-based topic modeler for aspect discovery from reviews.
    
    Features:
    - Automatic topic extraction from review text
    - Topic filtering by product category
    - Representative document extraction
    - Topic label generation
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        text_column: str = 'review',
        min_topic_size: int = 50,
        n_topics: Optional[int] = None
    ):
        """
        Initialize TopicModeler.
        
        Args:
            df: DataFrame with reviews
            text_column: Column containing review text
            min_topic_size: Minimum documents per topic
            n_topics: Number of topics (None for auto)
        """
        self.df = df.copy()
        self.text_column = text_column
        self.min_topic_size = min_topic_size
        self.n_topics = n_topics
        
        self.model = None
        self.topics = None
        self.topic_info = None
        self.embeddings = None
        
    def fit(self, sample_size: Optional[int] = None) -> 'TopicModeler':
        """
        Fit BERTopic model on reviews.
        
        Args:
            sample_size: Optional sample size for faster training
            
        Returns:
            Self for chaining
        """
        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error("BERTopic not installed. Run: pip install bertopic sentence-transformers")
            return self
            
        logger.info("Fitting BERTopic model...")
        
        # Get valid reviews
        valid_mask = self.df[self.text_column].notna() & (self.df[self.text_column] != '')
        docs = self.df.loc[valid_mask, self.text_column].astype(str).tolist()
        
        if sample_size and len(docs) > sample_size:
            import random
            indices = random.sample(range(len(docs)), sample_size)
            docs = [docs[i] for i in indices]
            logger.info(f"Sampled {sample_size} documents for training")
        
        logger.info(f"Training on {len(docs)} documents...")
        
        # Create embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create BERTopic model
        self.model = BERTopic(
            embedding_model=embedding_model,
            min_topic_size=self.min_topic_size,
            nr_topics=self.n_topics,
            verbose=True,
            calculate_probabilities=False
        )
        
        # Fit model
        self.topics, _ = self.model.fit_transform(docs)
        self.topic_info = self.model.get_topic_info()
        
        logger.info(f"Found {len(self.topic_info) - 1} topics (excluding outliers)")
        
        return self
        
    def get_topics(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top N topics with their keywords.
        
        Args:
            top_n: Number of topics to return
            
        Returns:
            DataFrame with topic information
        """
        if self.topic_info is None:
            logger.warning("Model not fitted. Call fit() first.")
            return pd.DataFrame()
            
        # Exclude outlier topic (-1)
        topics = self.topic_info[self.topic_info['Topic'] != -1].head(top_n)
        return topics
        
    def get_topic_keywords(self, topic_id: int, top_n: int = 10) -> List[str]:
        """
        Get keywords for a specific topic.
        
        Args:
            topic_id: Topic ID
            top_n: Number of keywords
            
        Returns:
            List of keywords
        """
        if self.model is None:
            return []
            
        topic_words = self.model.get_topic(topic_id)
        if topic_words:
            return [word for word, _ in topic_words[:top_n]]
        return []
        
    def get_topic_label(self, topic_id: int) -> str:
        """
        Get human-readable label for a topic.
        
        Args:
            topic_id: Topic ID
            
        Returns:
            Topic label string
        """
        keywords = self.get_topic_keywords(topic_id, top_n=3)
        return ' / '.join(keywords) if keywords else f"Topic {topic_id}"
        
    def get_representative_docs(self, topic_id: int, n_docs: int = 5) -> List[str]:
        """
        Get representative documents for a topic.
        
        Args:
            topic_id: Topic ID
            n_docs: Number of documents
            
        Returns:
            List of representative documents
        """
        if self.model is None:
            return []
            
        try:
            docs = self.model.get_representative_docs(topic_id)
            return docs[:n_docs] if docs else []
        except Exception:
            return []
            
    def get_topics_for_category(
        self,
        category: str,
        category_column: str = 'product_category',
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get top topics for a specific product category.
        
        Args:
            category: Product category name
            category_column: Column containing category
            top_n: Number of topics to return
            
        Returns:
            List of topic dictionaries
        """
        if self.model is None or self.topics is None:
            logger.warning("Model not fitted")
            return []
            
        # Filter to category
        valid_mask = self.df[self.text_column].notna() & (self.df[self.text_column] != '')
        filtered_df = self.df[valid_mask].copy()
        
        if category_column not in filtered_df.columns:
            logger.warning(f"Column '{category_column}' not found")
            return []
            
        category_mask = filtered_df[category_column] == category
        
        if category_mask.sum() == 0:
            logger.warning(f"No documents found for category: {category}")
            return []
            
        # Get topic distribution for this category
        category_indices = filtered_df[category_mask].index.tolist()
        
        # Map to topic assignments
        topic_counts = {}
        for idx in category_indices:
            if idx < len(self.topics):
                topic = self.topics[idx]
                if topic != -1:  # Exclude outliers
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
                    
        # Sort by count
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for topic_id, count in sorted_topics[:top_n]:
            results.append({
                'topic_id': topic_id,
                'label': self.get_topic_label(topic_id),
                'keywords': self.get_topic_keywords(topic_id),
                'count': count,
                'representative_docs': self.get_representative_docs(topic_id, 3)
            })
            
        return results
        
    def find_topic_by_aspect(self, aspect: str) -> List[Dict[str, Any]]:
        """
        Find topics matching a given aspect/keyword.
        
        Args:
            aspect: Aspect or keyword to search
            
        Returns:
            List of matching topics
        """
        if self.model is None:
            return []
            
        aspect_lower = aspect.lower()
        matching_topics = []
        
        for topic_id in range(len(self.model.get_topics())):
            if topic_id == -1:
                continue
                
            keywords = self.get_topic_keywords(topic_id)
            # Check if aspect matches any keyword
            if any(aspect_lower in kw.lower() or kw.lower() in aspect_lower for kw in keywords):
                matching_topics.append({
                    'topic_id': topic_id,
                    'label': self.get_topic_label(topic_id),
                    'keywords': keywords,
                    'relevance': sum(1 for kw in keywords if aspect_lower in kw.lower())
                })
                
        # Sort by relevance
        matching_topics.sort(key=lambda x: x['relevance'], reverse=True)
        return matching_topics
        
    def get_documents_for_aspect(
        self,
        aspect: str,
        max_docs: int = 100
    ) -> pd.DataFrame:
        """
        Get documents related to a specific aspect.
        
        Args:
            aspect: Aspect to search for
            max_docs: Maximum documents to return
            
        Returns:
            DataFrame with matching documents
        """
        # First try to find matching topics
        matching = self.find_topic_by_aspect(aspect)
        
        if matching and self.topics is not None:
            # Get documents from matching topics
            topic_ids = [t['topic_id'] for t in matching[:3]]  # Top 3 matching topics
            
            valid_mask = self.df[self.text_column].notna() & (self.df[self.text_column] != '')
            filtered_df = self.df[valid_mask].copy()
            
            # Filter to matching topics
            doc_mask = [self.topics[i] in topic_ids for i in range(min(len(self.topics), len(filtered_df)))]
            result_df = filtered_df.iloc[:len(doc_mask)][doc_mask]
            
            return result_df.head(max_docs)
            
        # Fallback: keyword search
        valid_mask = self.df[self.text_column].notna()
        filtered_df = self.df[valid_mask]
        
        keyword_mask = filtered_df[self.text_column].str.lower().str.contains(aspect.lower(), na=False)
        return filtered_df[keyword_mask].head(max_docs)
        
    def save(self, path: Path) -> None:
        """Save model to disk."""
        if self.model is None:
            logger.warning("No model to save")
            return
            
        self.model.save(str(path))
        logger.info(f"Model saved to {path}")
        
    def load(self, path: Path) -> 'TopicModeler':
        """Load model from disk."""
        try:
            from bertopic import BERTopic
            self.model = BERTopic.load(str(path))
            self.topic_info = self.model.get_topic_info()
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
        return self
        
    def get_category_aspect_summary(
        self,
        category: str,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Get summary of top aspects for a category.
        
        Args:
            category: Product category
            top_n: Number of top aspects
            
        Returns:
            Summary dictionary
        """
        topics = self.get_topics_for_category(category, top_n=top_n)
        
        return {
            'category': category,
            'total_aspects': len(topics),
            'aspects': [
                {
                    'name': t['label'],
                    'keywords': t['keywords'][:5],
                    'mention_count': t['count'],
                    'sample_reviews': t['representative_docs'][:2]
                }
                for t in topics
            ]
        }
