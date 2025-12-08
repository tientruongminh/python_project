"""
Aspect Summarizer using LLM for generating natural language summaries.

Combines BERTopic topics with Gemini LLM to create readable summaries
of user feedback for specific aspects or product categories.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

import pandas as pd

from src.config.settings import settings
from src.clustering.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class AspectSummarizer:
    """
    LLM-based summarizer for aspect-based review summarization.
    
    Features:
    - Summarize reviews for a specific aspect
    - Get top aspects summary for a category
    - Generate actionable insights
    """
    
    def __init__(self, df: pd.DataFrame, topic_modeler=None):
        """
        Initialize AspectSummarizer.
        
        Args:
            df: DataFrame with reviews
            topic_modeler: Optional TopicModeler instance
        """
        self.df = df.copy()
        self.topic_modeler = topic_modeler
        self.gemini_client = None
        
        self._init_gemini()
        
    def _init_gemini(self) -> None:
        """Initialize Gemini client."""
        try:
            self.gemini_client = GeminiClient()
            if not self.gemini_client.is_available:
                logger.warning("Gemini API not available")
                self.gemini_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini: {e}")
            self.gemini_client = None
            
    def summarize_aspect(
        self,
        aspect: str,
        category: Optional[str] = None,
        max_reviews: int = 50
    ) -> Dict[str, Any]:
        """
        Summarize what users say about a specific aspect.
        
        Args:
            aspect: Aspect name (e.g., "battery life", "sound quality")
            category: Optional product category to filter
            max_reviews: Maximum reviews to analyze
            
        Returns:
            Summary dictionary with insights
        """
        logger.info(f"Summarizing aspect: {aspect}")
        
        # Get relevant reviews
        reviews = self._get_reviews_for_aspect(aspect, category, max_reviews)
        
        if len(reviews) == 0:
            return {
                'aspect': aspect,
                'category': category,
                'review_count': 0,
                'summary': f"No reviews found mentioning '{aspect}'.",
                'sentiment': 'neutral',
                'key_points': []
            }
            
        # Analyze sentiment distribution
        sentiment_dist = self._analyze_sentiment(reviews)
        
        # Generate LLM summary
        summary = self._generate_summary(aspect, reviews, category)
        
        # Extract key points
        key_points = self._extract_key_points(aspect, reviews)
        
        return {
            'aspect': aspect,
            'category': category,
            'review_count': len(reviews),
            'summary': summary,
            'sentiment': sentiment_dist['dominant'],
            'sentiment_scores': sentiment_dist,
            'key_points': key_points,
            'sample_reviews': reviews[:5].tolist()
        }
        
    def get_top_aspects_for_category(
        self,
        category: str,
        n_aspects: int = 5
    ) -> Dict[str, Any]:
        """
        Get top N aspects discussed for a product category.
        
        Args:
            category: Product category name
            n_aspects: Number of aspects to return
            
        Returns:
            Dictionary with top aspects and summaries
        """
        logger.info(f"Getting top {n_aspects} aspects for: {category}")
        
        # Filter to category
        if 'product_category' in self.df.columns:
            category_df = self.df[self.df['product_category'] == category]
        else:
            category_df = self.df
            
        if len(category_df) == 0:
            return {
                'category': category,
                'error': f"No reviews found for category: {category}",
                'aspects': []
            }
            
        # Get aspects from TopicModeler if available
        if self.topic_modeler is not None:
            topics = self.topic_modeler.get_topics_for_category(category, top_n=n_aspects)
            aspects = [t['label'] for t in topics]
        else:
            # Fallback: use predefined aspects
            aspects = self._get_common_aspects(category_df, n_aspects)
            
        # Summarize each aspect
        aspect_summaries = []
        for aspect in aspects[:n_aspects]:
            summary = self.summarize_aspect(aspect, category, max_reviews=30)
            aspect_summaries.append(summary)
            
        return {
            'category': category,
            'total_reviews': len(category_df),
            'aspects': aspect_summaries
        }
        
    def _get_reviews_for_aspect(
        self,
        aspect: str,
        category: Optional[str],
        max_reviews: int
    ) -> pd.Series:
        """Get reviews mentioning a specific aspect."""
        df = self.df.copy()
        
        # Filter by category if specified
        if category and 'product_category' in df.columns:
            df = df[df['product_category'] == category]
            
        # Filter to valid reviews
        if 'review' not in df.columns:
            return pd.Series(dtype=str)
            
        valid_mask = df['review'].notna() & (df['review'] != '')
        df = df[valid_mask]
        
        # Search for aspect in reviews
        aspect_lower = aspect.lower()
        keywords = aspect_lower.split()
        
        # Match any keyword
        mask = df['review'].str.lower().str.contains('|'.join(keywords), na=False, regex=True)
        
        matching = df[mask]['review'].head(max_reviews)
        return matching
        
    def _analyze_sentiment(self, reviews: pd.Series) -> Dict[str, Any]:
        """Analyze sentiment distribution of reviews."""
        positive_keywords = ['great', 'excellent', 'love', 'amazing', 'good', 'best', 'perfect', 'happy']
        negative_keywords = ['bad', 'terrible', 'hate', 'awful', 'worst', 'poor', 'broken', 'disappointed']
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for review in reviews:
            review_lower = str(review).lower()
            has_positive = any(kw in review_lower for kw in positive_keywords)
            has_negative = any(kw in review_lower for kw in negative_keywords)
            
            if has_positive and not has_negative:
                positive_count += 1
            elif has_negative and not has_positive:
                negative_count += 1
            else:
                neutral_count += 1
                
        total = len(reviews)
        if total == 0:
            return {'positive': 0, 'negative': 0, 'neutral': 0, 'dominant': 'neutral'}
            
        result = {
            'positive': round(positive_count / total * 100, 1),
            'negative': round(negative_count / total * 100, 1),
            'neutral': round(neutral_count / total * 100, 1)
        }
        
        # Determine dominant sentiment
        max_sentiment = max(result, key=result.get)
        result['dominant'] = max_sentiment
        
        return result
        
    def _generate_summary(
        self,
        aspect: str,
        reviews: pd.Series,
        category: Optional[str]
    ) -> str:
        """Generate LLM summary of reviews."""
        if self.gemini_client is None:
            return self._generate_fallback_summary(aspect, reviews)
            
        # Prepare sample reviews for prompt
        sample_reviews = reviews.head(20).tolist()
        reviews_text = '\n'.join([f"- {r[:200]}..." if len(str(r)) > 200 else f"- {r}" for r in sample_reviews])
        
        category_context = f" for {category} products" if category else ""
        
        prompt = f"""Analyze these customer reviews about "{aspect}"{category_context}.

Reviews:
{reviews_text}

Provide a concise summary (2-3 sentences) of what customers think about {aspect}. 
Focus on:
1. Overall sentiment (positive/negative/mixed)
2. Common praises or complaints
3. Key takeaways

Summary:"""

        try:
            summary = self.gemini_client.generate(prompt, max_tokens=200)
            return summary.strip() if summary else self._generate_fallback_summary(aspect, reviews)
        except Exception as e:
            logger.warning(f"LLM summary failed: {e}")
            return self._generate_fallback_summary(aspect, reviews)
            
    def _generate_fallback_summary(self, aspect: str, reviews: pd.Series) -> str:
        """Generate simple summary without LLM."""
        sentiment = self._analyze_sentiment(reviews)
        
        if sentiment['positive'] > sentiment['negative']:
            tone = "generally positive"
        elif sentiment['negative'] > sentiment['positive']:
            tone = "generally negative"
        else:
            tone = "mixed"
            
        return (
            f"Based on {len(reviews)} reviews mentioning '{aspect}', "
            f"customer feedback is {tone}. "
            f"Positive: {sentiment['positive']}%, "
            f"Negative: {sentiment['negative']}%, "
            f"Neutral: {sentiment['neutral']}%."
        )
        
    def _extract_key_points(self, aspect: str, reviews: pd.Series, n_points: int = 5) -> List[str]:
        """Extract key points from reviews."""
        if self.gemini_client is None:
            return []
            
        sample_reviews = reviews.head(15).tolist()
        reviews_text = '\n'.join([f"- {str(r)[:150]}" for r in sample_reviews])
        
        prompt = f"""From these reviews about "{aspect}", extract {n_points} key points.

Reviews:
{reviews_text}

List {n_points} brief key points (one line each):"""

        try:
            response = self.gemini_client.generate(prompt, max_tokens=300)
            if response:
                # Parse bullet points
                lines = response.strip().split('\n')
                points = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Remove bullet markers
                        line = line.lstrip('•-*0123456789. ')
                        if line:
                            points.append(line)
                return points[:n_points]
        except Exception as e:
            logger.warning(f"Key points extraction failed: {e}")
            
        return []
        
    def _get_common_aspects(self, df: pd.DataFrame, n: int) -> List[str]:
        """Get common aspects from reviews using keyword matching."""
        common_aspects = [
            'quality', 'price', 'value', 'shipping', 'delivery',
            'sound', 'picture', 'battery', 'screen', 'size',
            'easy to use', 'setup', 'customer service', 'warranty',
            'design', 'durability', 'performance', 'features'
        ]
        
        if 'review' not in df.columns:
            return common_aspects[:n]
            
        # Count mentions
        aspect_counts = {}
        reviews = df['review'].dropna().str.lower()
        
        for aspect in common_aspects:
            count = reviews.str.contains(aspect, na=False).sum()
            if count > 0:
                aspect_counts[aspect] = count
                
        # Sort by count
        sorted_aspects = sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True)
        return [a[0] for a in sorted_aspects[:n]]
        
    def generate_category_report(self, category: str) -> str:
        """
        Generate a full text report for a category.
        
        Args:
            category: Product category
            
        Returns:
            Markdown report string
        """
        result = self.get_top_aspects_for_category(category, n_aspects=5)
        
        if 'error' in result:
            return f"Error: {result['error']}"
            
        report_lines = [
            f"# {category} - Review Analysis Report",
            "",
            f"Total Reviews Analyzed: {result['total_reviews']}",
            "",
            "## Top Aspects Discussed",
            ""
        ]
        
        for i, aspect in enumerate(result['aspects'], 1):
            report_lines.extend([
                f"### {i}. {aspect['aspect']}",
                f"",
                f"**Reviews mentioning this aspect:** {aspect['review_count']}",
                f"",
                f"**Sentiment:** {aspect['sentiment']} "
                f"(Positive: {aspect.get('sentiment_scores', {}).get('positive', 0)}%, "
                f"Negative: {aspect.get('sentiment_scores', {}).get('negative', 0)}%)",
                f"",
                f"**Summary:** {aspect['summary']}",
                ""
            ])
            
            if aspect.get('key_points'):
                report_lines.append("**Key Points:**")
                for point in aspect['key_points']:
                    report_lines.append(f"- {point}")
                report_lines.append("")
                
        return '\n'.join(report_lines)
