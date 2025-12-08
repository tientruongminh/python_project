"""
Aspect extractor for review analysis.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

import pandas as pd

from src.config.settings import settings

logger = logging.getLogger(__name__)


class AspectExtractor:
    """
    Extract aspects from product reviews.
    
    Uses keyword-based and pattern-based extraction.
    """
    
    # Aspect keyword mappings
    ASPECT_KEYWORDS = {
        'quality': [
            'quality', 'well made', 'well-made', 'durable', 'sturdy', 
            'build', 'construction', 'craftsmanship', 'material'
        ],
        'price': [
            'price', 'cost', 'expensive', 'cheap', 'affordable', 
            'value', 'worth', 'money', 'budget', 'overpriced'
        ],
        'shipping': [
            'shipping', 'delivery', 'arrived', 'package', 'shipped',
            'fedex', 'ups', 'usps', 'tracking', 'fast', 'slow'
        ],
        'packaging': [
            'packaging', 'box', 'packed', 'wrapped', 'damaged',
            'broken', 'dented', 'crushed'
        ],
        'durability': [
            'durable', 'lasted', 'broke', 'worn', 'tear',
            'longevity', 'lifespan', 'withstand'
        ],
        'ease_of_use': [
            'easy', 'difficult', 'simple', 'complicated', 'intuitive',
            'user-friendly', 'setup', 'install', 'assemble'
        ],
        'appearance': [
            'look', 'looks', 'beautiful', 'ugly', 'design',
            'aesthetic', 'color', 'style', 'appearance'
        ],
        'size': [
            'size', 'fit', 'small', 'large', 'too big',
            'too small', 'perfect fit', 'sizing'
        ],
        'customer_service': [
            'customer service', 'support', 'return', 'refund',
            'warranty', 'replacement', 'response'
        ],
        'functionality': [
            'works', 'function', 'features', 'performance',
            'effective', 'efficient', 'powerful'
        ]
    }
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize aspect extractor.
        
        Args:
            df: DataFrame with review data
        """
        self.df = df.copy()
        self.config = settings
        
    def extract_aspects(self) -> pd.DataFrame:
        """
        Extract aspects from all reviews.
        
        Returns:
            DataFrame with aspect columns added
        """
        logger.info("Extracting aspects from reviews...")
        
        if 'review' not in self.df.columns:
            logger.warning("No 'review' column found")
            return self.df
            
        # Initialize aspect columns
        for aspect in self.ASPECT_KEYWORDS.keys():
            self.df[f'has_{aspect}'] = False
            self.df[f'{aspect}_sentiment'] = None
            
        # Extract aspects for each review
        for idx, row in self.df.iterrows():
            review = row.get('review')
            if not review or not isinstance(review, str):
                continue
                
            aspects = self._extract_from_text(review)
            
            for aspect, sentiment in aspects.items():
                self.df.at[idx, f'has_{aspect}'] = True
                self.df.at[idx, f'{aspect}_sentiment'] = sentiment
                
        # Log extraction stats
        for aspect in self.ASPECT_KEYWORDS.keys():
            count = self.df[f'has_{aspect}'].sum()
            logger.info(f"Found {count} reviews mentioning '{aspect}'")
            
        return self.df
        
    def _extract_from_text(self, text: str) -> Dict[str, str]:
        """
        Extract aspects and sentiments from text.
        
        Args:
            text: Review text
            
        Returns:
            Dict mapping aspect to sentiment
        """
        text_lower = text.lower()
        aspects = {}
        
        for aspect, keywords in self.ASPECT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Found aspect, determine sentiment
                    sentiment = self._determine_sentiment(text_lower, keyword)
                    aspects[aspect] = sentiment
                    break  # Found this aspect, move to next
                    
        return aspects
        
    def _determine_sentiment(self, text: str, keyword: str) -> str:
        """
        Determine sentiment around a keyword.
        
        Args:
            text: Full review text (lowercase)
            keyword: Keyword found in text
            
        Returns:
            'positive', 'negative', or 'neutral'
        """
        positive_words = self.config.analysis.positive_keywords
        negative_words = self.config.analysis.negative_keywords
        
        # Get context around keyword (50 chars before and after)
        idx = text.find(keyword)
        start = max(0, idx - 50)
        end = min(len(text), idx + len(keyword) + 50)
        context = text[start:end]
        
        # Count sentiment words in context
        pos_count = sum(1 for word in positive_words if word in context)
        neg_count = sum(1 for word in negative_words if word in context)
        
        # Check for negation
        negation_words = ['not', "n't", 'never', 'no', 'none', 'nothing']
        has_negation = any(neg in context for neg in negation_words)
        
        if pos_count > neg_count:
            return 'negative' if has_negation else 'positive'
        elif neg_count > pos_count:
            return 'positive' if has_negation else 'negative'
        else:
            return 'neutral'
            
    def get_aspect_summary(self, group_by: Optional[str] = None) -> pd.DataFrame:
        """
        Get summary of aspect mentions and sentiments.
        
        Args:
            group_by: Optional column to group by (e.g., 'product_category')
            
        Returns:
            DataFrame with aspect statistics
        """
        aspects = list(self.ASPECT_KEYWORDS.keys())
        
        if group_by and group_by in self.df.columns:
            # Group by specified column
            summaries = []
            
            for group_val in self.df[group_by].unique():
                group_df = self.df[self.df[group_by] == group_val]
                
                for aspect in aspects:
                    has_col = f'has_{aspect}'
                    sent_col = f'{aspect}_sentiment'
                    
                    if has_col not in self.df.columns:
                        continue
                        
                    mentions = group_df[has_col].sum()
                    total = len(group_df)
                    
                    # Sentiment breakdown
                    pos = (group_df[sent_col] == 'positive').sum()
                    neg = (group_df[sent_col] == 'negative').sum()
                    neu = (group_df[sent_col] == 'neutral').sum()
                    
                    summaries.append({
                        'group': group_val,
                        'aspect': aspect,
                        'mentions': mentions,
                        'mention_rate': round(mentions / total * 100, 2) if total > 0 else 0,
                        'positive': pos,
                        'negative': neg,
                        'neutral': neu,
                        'sentiment_score': round((pos - neg) / mentions * 100, 2) if mentions > 0 else 0
                    })
                    
            return pd.DataFrame(summaries)
            
        else:
            # Overall summary
            summaries = []
            
            for aspect in aspects:
                has_col = f'has_{aspect}'
                sent_col = f'{aspect}_sentiment'
                
                if has_col not in self.df.columns:
                    continue
                    
                mentions = self.df[has_col].sum()
                total = len(self.df)
                
                pos = (self.df[sent_col] == 'positive').sum()
                neg = (self.df[sent_col] == 'negative').sum()
                neu = (self.df[sent_col] == 'neutral').sum()
                
                summaries.append({
                    'aspect': aspect,
                    'mentions': mentions,
                    'mention_rate': round(mentions / total * 100, 2) if total > 0 else 0,
                    'positive': pos,
                    'negative': neg,
                    'neutral': neu,
                    'sentiment_score': round((pos - neg) / mentions * 100, 2) if mentions > 0 else 0
                })
                
            return pd.DataFrame(summaries).sort_values('mentions', ascending=False)
