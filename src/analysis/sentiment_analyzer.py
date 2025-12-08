"""
Sentiment analyzer for product reviews.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import pandas as pd

from src.config.settings import settings

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Analyze sentiment of product reviews.
    
    Provides both overall sentiment and aspect-based sentiment.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize sentiment analyzer.
        
        Args:
            df: DataFrame with review data
        """
        self.df = df.copy()
        self.config = settings
        
    def analyze_overall_sentiment(self) -> pd.DataFrame:
        """
        Analyze overall sentiment of each review.
        
        Returns:
            DataFrame with 'sentiment' and 'sentiment_score' columns
        """
        logger.info("Analyzing overall sentiment...")
        
        if 'review' not in self.df.columns:
            logger.warning("No 'review' column found")
            return self.df
            
        self.df['sentiment_score'] = self.df['review'].apply(self._calculate_sentiment_score)
        self.df['sentiment'] = self.df['sentiment_score'].apply(self._score_to_label)
        
        # Log sentiment distribution
        sentiment_dist = self.df['sentiment'].value_counts()
        logger.info(f"Sentiment distribution: {sentiment_dist.to_dict()}")
        
        return self.df
        
    def _calculate_sentiment_score(self, text: str) -> float:
        """
        Calculate sentiment score for text.
        
        Args:
            text: Review text
            
        Returns:
            Score from -1.0 (negative) to 1.0 (positive)
        """
        if not text or not isinstance(text, str):
            return 0.0
            
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_words = set(self.config.analysis.positive_keywords)
        negative_words = set(self.config.analysis.negative_keywords)
        
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
            
        score = (pos_count - neg_count) / total
        return round(score, 3)
        
    def _score_to_label(self, score: float) -> str:
        """Convert sentiment score to label."""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
            
    def analyze_by_category(self) -> pd.DataFrame:
        """
        Analyze sentiment by product category.
        
        Returns:
            DataFrame with category sentiment statistics
        """
        if 'product_category' not in self.df.columns:
            logger.warning("No 'product_category' column found")
            return pd.DataFrame()
            
        if 'sentiment' not in self.df.columns:
            self.analyze_overall_sentiment()
            
        summary = self.df.groupby('product_category').agg({
            'sentiment_score': ['mean', 'std'],
            'rating': 'mean',
            'uniq_id': 'count'
        }).round(3)
        
        summary.columns = ['avg_sentiment', 'sentiment_std', 'avg_rating', 'review_count']
        summary = summary.sort_values('review_count', ascending=False)
        
        return summary
        
    def analyze_by_time(self, date_column: str = 'review_date') -> pd.DataFrame:
        """
        Analyze sentiment trends over time.
        
        Args:
            date_column: Name of date column
            
        Returns:
            DataFrame with time-based sentiment statistics
        """
        if date_column not in self.df.columns:
            logger.warning(f"No '{date_column}' column found")
            return pd.DataFrame()
            
        if 'sentiment_score' not in self.df.columns:
            self.analyze_overall_sentiment()
            
        # Ensure datetime
        df_temp = self.df.copy()
        df_temp[date_column] = pd.to_datetime(df_temp[date_column], errors='coerce')
        
        # Remove null dates
        df_temp = df_temp.dropna(subset=[date_column])
        
        if len(df_temp) == 0:
            return pd.DataFrame()
            
        # Group by month
        df_temp['year_month'] = df_temp[date_column].dt.to_period('M')
        
        summary = df_temp.groupby('year_month').agg({
            'sentiment_score': 'mean',
            'rating': 'mean',
            'uniq_id': 'count'
        }).round(3)
        
        summary.columns = ['avg_sentiment', 'avg_rating', 'review_count']
        
        return summary
        
    def get_sentiment_drivers(
        self,
        sentiment_type: str = 'negative',
        category: Optional[str] = None,
        top_n: int = 10
    ) -> List[Dict[str, any]]:
        """
        Find common themes in reviews of specific sentiment.
        
        Args:
            sentiment_type: 'positive', 'negative', or 'neutral'
            category: Optional category to filter
            top_n: Number of top themes to return
            
        Returns:
            List of theme dictionaries
        """
        if 'sentiment' not in self.df.columns:
            self.analyze_overall_sentiment()
            
        # Filter by sentiment
        filtered = self.df[self.df['sentiment'] == sentiment_type]
        
        # Filter by category if specified
        if category and 'product_category' in self.df.columns:
            filtered = filtered[filtered['product_category'] == category]
            
        if len(filtered) == 0:
            return []
            
        # Extract common words/phrases
        from collections import Counter
        
        all_words = []
        for review in filtered['review'].dropna():
            words = str(review).lower().split()
            # Filter stop words and short words
            stop_words = {'the', 'a', 'an', 'is', 'it', 'was', 'were', 'be', 'been',
                         'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                         'shall', 'would', 'could', 'should', 'may', 'might', 'must',
                         'and', 'but', 'or', 'nor', 'for', 'yet', 'so', 'to', 'of',
                         'in', 'on', 'at', 'by', 'with', 'from', 'this', 'that', 'these',
                         'those', 'i', 'you', 'he', 'she', 'we', 'they', 'my', 'your'}
            words = [w for w in words if w not in stop_words and len(w) > 2]
            all_words.extend(words)
            
        word_counts = Counter(all_words).most_common(top_n)
        
        themes = []
        for word, count in word_counts:
            themes.append({
                'theme': word,
                'count': count,
                'percentage': round(count / len(filtered) * 100, 2)
            })
            
        return themes
