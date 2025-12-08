"""
Product clusterer module using Gemini API.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Optional, Any
from collections import Counter

import pandas as pd

from src.config.settings import settings
from src.clustering.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class ProductClusterer:
    """
    Cluster products into categories using Gemini API.
    
    Handles:
    - Product grouping by title/URL
    - Category assignment via LLM
    - Category consolidation
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize clusterer with DataFrame.
        
        Args:
            df: DataFrame with product data
        """
        self.df = df.copy()
        self.config = settings
        self.gemini_client = GeminiClient()
        self.category_cache: Dict[str, str] = {}
        
    def cluster_products(self) -> pd.DataFrame:
        """
        Cluster products and assign categories.
        
        Returns:
            DataFrame with 'product_category' column added
        """
        logger.info("Starting product clustering...")
        
        if not self.gemini_client.is_available:
            logger.warning("Gemini not available, using fallback clustering")
            return self._fallback_clustering()
            
        # Get unique products
        unique_products = self._get_unique_products()
        logger.info(f"Found {len(unique_products)} unique products to categorize")
        
        # Categorize products
        categorized = self.gemini_client.categorize_products(unique_products)
        
        # Build category map
        for cat_info in categorized:
            title = cat_info.get('title', '')
            category = cat_info.get('category', 'Unknown')
            self.category_cache[title] = category
            
        # Apply categories to DataFrame
        self.df['product_category'] = self.df['title'].map(
            lambda x: self.category_cache.get(x, 'Unknown')
        )
        
        # Log category distribution
        category_counts = self.df['product_category'].value_counts()
        logger.info(f"Created {len(category_counts)} categories")
        for cat, count in category_counts.head(10).items():
            logger.info(f"  {cat}: {count} reviews")
            
        return self.df
        
    def _get_unique_products(self) -> List[Dict[str, str]]:
        """
        Get unique products for categorization.
        
        Returns:
            List of unique product info dicts
        """
        if 'title' not in self.df.columns:
            logger.warning("No 'title' column found")
            return []
            
        # Get unique titles with context
        unique_titles = self.df['title'].dropna().unique()
        
        products = []
        for title in unique_titles:
            # Get a sample row for context
            sample_row = self.df[self.df['title'] == title].iloc[0]
            
            product_info = {'title': str(title)}
            
            # Add URL if available
            if 'pageurl' in sample_row:
                product_info['url'] = str(sample_row['pageurl'])
                
            # Add sample review for context
            if 'review' in sample_row and pd.notna(sample_row['review']):
                product_info['review'] = str(sample_row['review'])[:200]
                
            products.append(product_info)
            
        return products
        
    def _fallback_clustering(self) -> pd.DataFrame:
        """
        Simple fallback clustering based on keywords.
        
        Returns:
            DataFrame with basic category assignment
        """
        logger.info("Using keyword-based fallback clustering...")
        
        # Define keyword-to-category mappings
        category_keywords = {
            'Electronics': ['phone', 'laptop', 'tablet', 'computer', 'camera', 'headphone', 'speaker', 'tv', 'electronic'],
            'Home & Kitchen': ['kitchen', 'cookware', 'utensil', 'appliance', 'furniture', 'home', 'decor', 'bedding'],
            'Clothing & Fashion': ['shirt', 'pants', 'dress', 'shoes', 'clothing', 'fashion', 'wear', 'apparel'],
            'Toys & Games': ['toy', 'game', 'puzzle', 'lego', 'doll', 'action figure', 'kids'],
            'Health & Beauty': ['health', 'beauty', 'skincare', 'makeup', 'vitamin', 'supplement', 'personal care'],
            'Sports & Outdoors': ['sport', 'fitness', 'outdoor', 'camping', 'exercise', 'gym', 'bike'],
            'Baby & Kids': ['baby', 'infant', 'toddler', 'diaper', 'stroller', 'child'],
            'Food & Grocery': ['food', 'snack', 'grocery', 'beverage', 'drink', 'organic'],
            'Pet Supplies': ['pet', 'dog', 'cat', 'animal', 'fish'],
            'Office & School': ['office', 'school', 'supplies', 'desk', 'pen', 'paper'],
        }
        
        def categorize_by_keywords(title: str) -> str:
            if not title or not isinstance(title, str):
                return 'Other'
                
            title_lower = title.lower()
            
            for category, keywords in category_keywords.items():
                if any(kw in title_lower for kw in keywords):
                    return category
                    
            return 'Other'
            
        self.df['product_category'] = self.df['title'].apply(categorize_by_keywords)
        
        return self.df
        
    def consolidate_categories(self, min_count: int = 10) -> pd.DataFrame:
        """
        Consolidate small categories into 'Other'.
        
        Args:
            min_count: Minimum reviews per category
            
        Returns:
            DataFrame with consolidated categories
        """
        if 'product_category' not in self.df.columns:
            return self.df
            
        category_counts = self.df['product_category'].value_counts()
        small_categories = category_counts[category_counts < min_count].index.tolist()
        
        if small_categories:
            logger.info(f"Consolidating {len(small_categories)} small categories to 'Other'")
            self.df.loc[
                self.df['product_category'].isin(small_categories),
                'product_category'
            ] = 'Other'
            
        return self.df
        
    def get_category_summary(self) -> pd.DataFrame:
        """
        Get summary statistics by category.
        
        Returns:
            DataFrame with category statistics
        """
        if 'product_category' not in self.df.columns:
            return pd.DataFrame()
            
        summary = self.df.groupby('product_category').agg({
            'uniq_id': 'count',
            'rating': 'mean',
            'review_upvotes': 'sum',
            'review_downvotes': 'sum'
        }).rename(columns={
            'uniq_id': 'review_count',
            'rating': 'avg_rating'
        })
        
        summary = summary.round(2).sort_values('review_count', ascending=False)
        
        return summary
