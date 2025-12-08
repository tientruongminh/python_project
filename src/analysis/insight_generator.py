"""
Insight generator for business recommendations.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd

from src.config.settings import settings, OUTPUT_DIR
from src.analysis.aspect_extractor import AspectExtractor
from src.analysis.sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)


class InsightGenerator:
    """
    Generate business insights and recommendations from analysis.
    
    Provides:
    - Category-specific insights
    - Aspect-based recommendations
    - Time trend analysis
    - Actionable suggestions
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize insight generator.
        
        Args:
            df: Fully processed DataFrame with analysis results
        """
        self.df = df.copy()
        self.config = settings
        self.insights: Dict[str, Any] = {}
        
    def generate_all_insights(self) -> Dict[str, Any]:
        """
        Generate all insights from the data.
        
        Returns:
            Dictionary with all insights
        """
        logger.info("Generating insights...")
        
        self.insights = {
            'generated_at': datetime.now().isoformat(),
            'data_summary': self._get_data_summary(),
            'category_insights': self._get_category_insights(),
            'aspect_insights': self._get_aspect_insights(),
            'time_insights': self._get_time_insights(),
            'recommendations': self._generate_recommendations()
        }
        
        return self.insights
        
    def _get_data_summary(self) -> Dict[str, Any]:
        """Get overall data summary."""
        summary = {
            'total_reviews': len(self.df),
            'unique_products': self.df['product_id'].nunique() if 'product_id' in self.df.columns else 0,
            'date_range': {
                'start': str(self.df['review_date'].min()) if 'review_date' in self.df.columns else None,
                'end': str(self.df['review_date'].max()) if 'review_date' in self.df.columns else None
            },
            'avg_rating': round(self.df['rating'].mean(), 2) if 'rating' in self.df.columns else None,
            'categories_count': self.df['product_category'].nunique() if 'product_category' in self.df.columns else 0
        }
        
        return summary
        
    def _get_category_insights(self) -> List[Dict[str, Any]]:
        """Generate insights by product category."""
        if 'product_category' not in self.df.columns:
            return []
            
        insights = []
        
        for category in self.df['product_category'].unique():
            cat_df = self.df[self.df['product_category'] == category]
            
            if len(cat_df) < 10:  # Skip small categories
                continue
                
            insight = {
                'category': category,
                'review_count': len(cat_df),
                'avg_rating': round(cat_df['rating'].mean(), 2) if 'rating' in cat_df.columns else None,
                'sentiment_breakdown': {},
                'top_aspects': [],
                'pain_points': []
            }
            
            # Sentiment breakdown
            if 'sentiment' in cat_df.columns:
                sentiment_counts = cat_df['sentiment'].value_counts()
                total = len(cat_df)
                insight['sentiment_breakdown'] = {
                    'positive': round(sentiment_counts.get('positive', 0) / total * 100, 1),
                    'neutral': round(sentiment_counts.get('neutral', 0) / total * 100, 1),
                    'negative': round(sentiment_counts.get('negative', 0) / total * 100, 1)
                }
                
            # Top mentioned aspects
            aspect_cols = [c for c in cat_df.columns if c.startswith('has_')]
            for col in aspect_cols:
                aspect = col.replace('has_', '')
                mention_rate = cat_df[col].mean() * 100
                if mention_rate > 5:  # At least 5% mention rate
                    insight['top_aspects'].append({
                        'aspect': aspect,
                        'mention_rate': round(mention_rate, 1)
                    })
                    
            insight['top_aspects'] = sorted(
                insight['top_aspects'], 
                key=lambda x: x['mention_rate'], 
                reverse=True
            )[:5]
            
            # Identify pain points (negative sentiment aspects)
            for col in aspect_cols:
                aspect = col.replace('has_', '')
                sent_col = f'{aspect}_sentiment'
                
                if sent_col in cat_df.columns:
                    aspect_df = cat_df[cat_df[col] == True]
                    if len(aspect_df) > 0:
                        neg_rate = (aspect_df[sent_col] == 'negative').mean() * 100
                        if neg_rate > 30:  # More than 30% negative
                            insight['pain_points'].append({
                                'aspect': aspect,
                                'negative_rate': round(neg_rate, 1),
                                'mention_count': len(aspect_df)
                            })
                            
            insight['pain_points'] = sorted(
                insight['pain_points'],
                key=lambda x: x['negative_rate'],
                reverse=True
            )
            
            insights.append(insight)
            
        return sorted(insights, key=lambda x: x['review_count'], reverse=True)
        
    def _get_aspect_insights(self) -> Dict[str, Any]:
        """Generate overall aspect insights."""
        aspect_cols = [c for c in self.df.columns if c.startswith('has_')]
        
        aspect_stats = []
        for col in aspect_cols:
            aspect = col.replace('has_', '')
            sent_col = f'{aspect}_sentiment'
            
            mention_count = self.df[col].sum()
            mention_rate = self.df[col].mean() * 100
            
            if sent_col in self.df.columns:
                aspect_df = self.df[self.df[col] == True]
                if len(aspect_df) > 0:
                    pos_rate = (aspect_df[sent_col] == 'positive').mean() * 100
                    neg_rate = (aspect_df[sent_col] == 'negative').mean() * 100
                else:
                    pos_rate = neg_rate = 0
            else:
                pos_rate = neg_rate = None
                
            aspect_stats.append({
                'aspect': aspect,
                'mention_count': int(mention_count),
                'mention_rate': round(mention_rate, 2),
                'positive_rate': round(pos_rate, 2) if pos_rate else None,
                'negative_rate': round(neg_rate, 2) if neg_rate else None
            })
            
        return {
            'aspects': sorted(aspect_stats, key=lambda x: x['mention_count'], reverse=True),
            'most_discussed': aspect_stats[0]['aspect'] if aspect_stats else None,
            'highest_positive': max(aspect_stats, key=lambda x: x['positive_rate'] or 0)['aspect'] if aspect_stats else None,
            'highest_negative': max(aspect_stats, key=lambda x: x['negative_rate'] or 0)['aspect'] if aspect_stats else None
        }
        
    def _get_time_insights(self) -> Dict[str, Any]:
        """Generate time-based insights."""
        if 'review_date' not in self.df.columns:
            return {}
            
        df_temp = self.df.copy()
        df_temp['review_date'] = pd.to_datetime(df_temp['review_date'], errors='coerce')
        df_temp = df_temp.dropna(subset=['review_date'])
        
        if len(df_temp) == 0:
            return {}
            
        df_temp['year_month'] = df_temp['review_date'].dt.to_period('M')
        
        monthly = df_temp.groupby('year_month').agg({
            'rating': 'mean',
            'sentiment_score': 'mean' if 'sentiment_score' in df_temp.columns else 'count',
            'uniq_id': 'count'
        }).round(2)
        
        trends = {
            'months_analyzed': len(monthly),
            'trend_direction': 'improving' if monthly['rating'].is_monotonic_increasing else 
                              'declining' if monthly['rating'].is_monotonic_decreasing else 'fluctuating',
            'peak_month': str(monthly['uniq_id'].idxmax()),
            'lowest_rating_month': str(monthly['rating'].idxmin()),
            'highest_rating_month': str(monthly['rating'].idxmax())
        }
        
        return trends
        
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Get aspect insights
        aspect_insights = self._get_aspect_insights()
        
        # Recommendation for most negative aspect
        if aspect_insights.get('highest_negative'):
            aspect = aspect_insights['highest_negative']
            recommendations.append({
                'priority': 'high',
                'area': aspect,
                'recommendation': f"Focus on improving {aspect}. This aspect has the highest negative sentiment rate.",
                'action': f"Review negative reviews mentioning {aspect} and identify specific issues to address."
            })
            
        # Category-specific recommendations
        cat_insights = self._get_category_insights()
        for cat in cat_insights[:3]:  # Top 3 categories
            if cat.get('pain_points'):
                top_pain = cat['pain_points'][0]
                recommendations.append({
                    'priority': 'medium',
                    'area': f"{cat['category']} - {top_pain['aspect']}",
                    'recommendation': f"Address {top_pain['aspect']} issues in {cat['category']} category.",
                    'action': f"{top_pain['negative_rate']}% of reviews mentioning {top_pain['aspect']} are negative."
                })
                
        # Time-based recommendations
        time_insights = self._get_time_insights()
        if time_insights.get('trend_direction') == 'declining':
            recommendations.append({
                'priority': 'high',
                'area': 'Overall Trend',
                'recommendation': "Rating trend is declining. Investigate recent changes that may have impacted quality.",
                'action': "Compare products and processes from high-rating vs low-rating periods."
            })
            
        return recommendations
        
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate markdown report with all insights.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Markdown report content
        """
        if not self.insights:
            self.generate_all_insights()
            
        report = self._build_report()
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
            
        return report
        
    def _build_report(self) -> str:
        """Build markdown report content."""
        lines = [
            "# Walmart Product Review Analysis Report",
            "",
            f"*Generated: {self.insights.get('generated_at', 'N/A')}*",
            "",
            "---",
            "",
            "## Executive Summary",
            ""
        ]
        
        # Data summary
        summary = self.insights.get('data_summary', {})
        lines.extend([
            f"- **Total Reviews Analyzed**: {summary.get('total_reviews', 0):,}",
            f"- **Unique Products**: {summary.get('unique_products', 0):,}",
            f"- **Product Categories**: {summary.get('categories_count', 0)}",
            f"- **Average Rating**: {summary.get('avg_rating', 'N/A')} / 5.0",
            f"- **Date Range**: {summary.get('date_range', {}).get('start', 'N/A')} to {summary.get('date_range', {}).get('end', 'N/A')}",
            "",
            "---",
            "",
            "## Key Recommendations",
            ""
        ])
        
        # Recommendations
        for rec in self.insights.get('recommendations', []):
            priority_label = "[HIGH]" if rec['priority'] == 'high' else "[MEDIUM]" if rec['priority'] == 'medium' else "[LOW]"
            lines.extend([
                f"### {priority_label} {rec['area']}",
                f"**Recommendation**: {rec['recommendation']}",
                f"",
                f"**Action**: {rec['action']}",
                ""
            ])
            
        # Category insights
        lines.extend([
            "---",
            "",
            "## Category Performance",
            ""
        ])
        
        cat_insights = self.insights.get('category_insights', [])[:10]
        if cat_insights:
            lines.append("| Category | Reviews | Avg Rating | Positive % | Negative % |")
            lines.append("|----------|---------|------------|------------|------------|")
            
            for cat in cat_insights:
                sent = cat.get('sentiment_breakdown', {})
                lines.append(
                    f"| {cat['category']} | {cat['review_count']:,} | "
                    f"{cat.get('avg_rating', 'N/A')} | "
                    f"{sent.get('positive', 'N/A')}% | "
                    f"{sent.get('negative', 'N/A')}% |"
                )
                
        # Aspect analysis
        lines.extend([
            "",
            "---",
            "",
            "## Aspect Analysis",
            "",
            "### Most Discussed Aspects",
            ""
        ])
        
        aspect_insights = self.insights.get('aspect_insights', {})
        aspects = aspect_insights.get('aspects', [])[:10]
        
        if aspects:
            lines.append("| Aspect | Mentions | Mention Rate | Positive % | Negative % |")
            lines.append("|--------|----------|--------------|------------|------------|")
            
            for asp in aspects:
                lines.append(
                    f"| {asp['aspect']} | {asp['mention_count']:,} | "
                    f"{asp['mention_rate']}% | "
                    f"{asp.get('positive_rate', 'N/A')}% | "
                    f"{asp.get('negative_rate', 'N/A')}% |"
                )
                
        # Time trends
        time_insights = self.insights.get('time_insights', {})
        if time_insights:
            lines.extend([
                "",
                "---",
                "",
                "## Trend Analysis",
                "",
                f"- **Months Analyzed**: {time_insights.get('months_analyzed', 'N/A')}",
                f"- **Overall Trend**: {time_insights.get('trend_direction', 'N/A').title()}",
                f"- **Peak Activity Month**: {time_insights.get('peak_month', 'N/A')}",
                f"- **Highest Rating Month**: {time_insights.get('highest_rating_month', 'N/A')}",
                f"- **Lowest Rating Month**: {time_insights.get('lowest_rating_month', 'N/A')}",
            ])
            
        lines.extend([
            "",
            "---",
            "",
            "*Report generated by Walmart Review Analysis Pipeline*"
        ])
        
        return "\n".join(lines)
