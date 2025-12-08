"""
Streamlit Dashboard for Walmart Product Review Analysis.

Features:
- Project Report Dashboard
- Chatbot with 2 modes:
  1. Aspect Query: Ask about specific aspect
  2. Top-N Aspects: Get popular aspects for a category
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import settings, OUTPUT_DIR
from src.analysis.aspect_summarizer import AspectSummarizer

# Page config
st.set_page_config(
    page_title="Walmart Review Analysis",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - no icons
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .bot-message {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load processed data."""
    data_path = OUTPUT_DIR / "sentiment_analysis.csv"
    if data_path.exists():
        return pd.read_csv(data_path)
    
    # Fallback to processed data
    data_path = settings.processed_data_path
    if data_path.exists():
        return pd.read_csv(data_path)
        
    return None


@st.cache_resource
def get_summarizer(df):
    """Get or create AspectSummarizer."""
    return AspectSummarizer(df)


def render_sidebar():
    """Render sidebar with navigation."""
    st.sidebar.markdown("## Navigation")
    
    page = st.sidebar.radio(
        "Select Page",
        ["Project Report", "Chatbot"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown(
        "Walmart Product Review Analysis Dashboard. "
        "Uses BERTopic and LLM for aspect-based summarization."
    )
    
    return page


def render_project_report(df):
    """Render project report dashboard."""
    st.markdown('<div class="main-header">Walmart Product Review Analysis Report</div>', unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", f"{len(df):,}")
    
    with col2:
        unique_products = df['pageurl'].nunique() if 'pageurl' in df.columns else 0
        st.metric("Unique Products", f"{unique_products:,}")
    
    with col3:
        categories = df['product_category'].nunique() if 'product_category' in df.columns else 0
        st.metric("Categories", f"{categories:,}")
    
    with col4:
        avg_rating = df['rating'].mean() if 'rating' in df.columns else 0
        st.metric("Avg Rating", f"{avg_rating:.2f}")
    
    st.markdown("---")
    
    # Two columns layout
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown('<div class="section-header">Rating Distribution</div>', unsafe_allow_html=True)
        if 'rating' in df.columns:
            rating_counts = df['rating'].value_counts().sort_index()
            fig = px.bar(
                x=rating_counts.index,
                y=rating_counts.values,
                labels={'x': 'Rating', 'y': 'Count'},
                color=rating_counts.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.markdown('<div class="section-header">Sentiment Distribution</div>', unsafe_allow_html=True)
        if 'rating_sentiment' in df.columns:
            sentiment_counts = df['rating_sentiment'].value_counts()
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                color=sentiment_counts.index,
                color_discrete_map={
                    'Positive': '#4CAF50',
                    'Neutral': '#FFC107',
                    'Negative': '#F44336'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Category Performance
    st.markdown('<div class="section-header">Top Categories by Review Count</div>', unsafe_allow_html=True)
    if 'product_category' in df.columns:
        category_stats = df.groupby('product_category').agg({
            'rating': ['count', 'mean']
        }).round(2)
        category_stats.columns = ['Review Count', 'Avg Rating']
        category_stats = category_stats.sort_values('Review Count', ascending=False).head(10)
        
        fig = px.bar(
            category_stats.reset_index(),
            x='product_category',
            y='Review Count',
            color='Avg Rating',
            color_continuous_scale='RdYlGn',
            labels={'product_category': 'Category'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Aspect Analysis
    st.markdown('<div class="section-header">Aspect Mention Analysis</div>', unsafe_allow_html=True)
    
    aspect_cols = [col for col in df.columns if col.startswith('aspect_') and col.endswith('_sentiment')]
    if aspect_cols:
        aspect_data = []
        for col in aspect_cols:
            aspect_name = col.replace('aspect_', '').replace('_sentiment', '')
            mentions = (df[col] != 'none').sum()
            if mentions > 0:
                aspect_data.append({
                    'Aspect': aspect_name,
                    'Mentions': mentions,
                    'Mention Rate': f"{mentions/len(df)*100:.1f}%"
                })
        
        if aspect_data:
            aspect_df = pd.DataFrame(aspect_data).sort_values('Mentions', ascending=False)
            st.dataframe(aspect_df, use_container_width=True, hide_index=True)
    
    # Time Trend
    st.markdown("---")
    st.markdown('<div class="section-header">Review Trend Over Time</div>', unsafe_allow_html=True)
    
    if 'review_year_month' in df.columns:
        trend = df.groupby('review_year_month').size().reset_index(name='count')
        trend = trend.sort_values('review_year_month').tail(24)  # Last 24 months
        
        fig = px.line(
            trend,
            x='review_year_month',
            y='count',
            labels={'review_year_month': 'Month', 'count': 'Reviews'}
        )
        st.plotly_chart(fig, use_container_width=True)


def render_chatbot(df, summarizer):
    """Render chatbot interface."""
    st.markdown('<div class="main-header">Review Analysis Chatbot</div>', unsafe_allow_html=True)
    
    # Mode selection
    mode = st.radio(
        "Select Query Mode",
        ["Mode 1: Ask about specific aspect", "Mode 2: Top aspects for category"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if mode == "Mode 1: Ask about specific aspect":
        render_mode1_aspect_query(df, summarizer)
    else:
        render_mode2_top_aspects(df, summarizer)


def render_mode1_aspect_query(df, summarizer):
    """Mode 1: Query specific aspect."""
    st.markdown("### Ask about a specific aspect")
    st.markdown("Enter an aspect (e.g., 'sound quality', 'battery life', 'price') to see what customers say about it.")
    
    # Category filter (optional)
    categories = ['All Categories']
    if 'product_category' in df.columns:
        categories += sorted(df['product_category'].dropna().unique().tolist())
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        aspect_input = st.text_input(
            "Enter aspect:",
            placeholder="e.g., sound quality, battery, price, shipping"
        )
    
    with col2:
        category_filter = st.selectbox("Filter by category:", categories)
    
    if st.button("Analyze Aspect", type="primary"):
        if aspect_input:
            with st.spinner("Analyzing reviews..."):
                category = None if category_filter == "All Categories" else category_filter
                result = summarizer.summarize_aspect(aspect_input, category=category)
                
                display_aspect_result(result)
        else:
            st.warning("Please enter an aspect to analyze.")


def render_mode2_top_aspects(df, summarizer):
    """Mode 2: Top N aspects for category."""
    st.markdown("### Top aspects for a product category")
    st.markdown("Select a category and number of aspects to discover the most discussed topics.")
    
    # Get categories
    categories = []
    if 'product_category' in df.columns:
        categories = sorted(df['product_category'].dropna().unique().tolist())
        # Exclude 'Unknown' and 'Other' if present
        categories = [c for c in categories if c not in ['Unknown', 'Other']]
    
    if not categories:
        st.warning("No product categories found in data.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_category = st.selectbox("Select product category:", categories)
    
    with col2:
        n_aspects = st.number_input(
            "Number of aspects:",
            min_value=1,
            max_value=10,
            value=3
        )
    
    if st.button("Get Top Aspects", type="primary"):
        with st.spinner(f"Finding top {n_aspects} aspects for {selected_category}..."):
            result = summarizer.get_top_aspects_for_category(selected_category, n_aspects=int(n_aspects))
            
            display_category_aspects(result)


def display_aspect_result(result):
    """Display single aspect analysis result."""
    st.markdown("---")
    st.markdown(f"### Analysis: {result['aspect']}")
    
    if result.get('category'):
        st.markdown(f"**Category:** {result['category']}")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Reviews Analyzed", result['review_count'])
    
    with col2:
        sentiment = result.get('sentiment', 'neutral').capitalize()
        st.metric("Overall Sentiment", sentiment)
    
    with col3:
        scores = result.get('sentiment_scores', {})
        positive = scores.get('positive', 0)
        st.metric("Positive Rate", f"{positive}%")
    
    # Summary
    st.markdown("#### Summary")
    st.markdown(result['summary'])
    
    # Key Points
    if result.get('key_points'):
        st.markdown("#### Key Points")
        for point in result['key_points']:
            st.markdown(f"- {point}")
    
    # Sample Reviews
    if result.get('sample_reviews'):
        with st.expander("View Sample Reviews"):
            for i, review in enumerate(result['sample_reviews'][:5], 1):
                st.markdown(f"**{i}.** {review[:300]}..." if len(str(review)) > 300 else f"**{i}.** {review}")


def display_category_aspects(result):
    """Display category aspects result."""
    st.markdown("---")
    st.markdown(f"### Top Aspects: {result['category']}")
    st.markdown(f"**Total Reviews in Category:** {result.get('total_reviews', 0):,}")
    
    if 'error' in result:
        st.error(result['error'])
        return
    
    aspects = result.get('aspects', [])
    
    if not aspects:
        st.info("No aspects found for this category.")
        return
    
    for i, aspect in enumerate(aspects, 1):
        with st.container():
            st.markdown(f"#### {i}. {aspect['aspect']}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mentions", aspect['review_count'])
            
            with col2:
                sentiment = aspect.get('sentiment', 'neutral').capitalize()
                st.metric("Sentiment", sentiment)
            
            with col3:
                scores = aspect.get('sentiment_scores', {})
                positive = scores.get('positive', 0)
                negative = scores.get('negative', 0)
                st.metric("Pos/Neg", f"{positive}% / {negative}%")
            
            st.markdown(f"**Summary:** {aspect['summary']}")
            
            if aspect.get('key_points'):
                with st.expander("Key Points"):
                    for point in aspect['key_points']:
                        st.markdown(f"- {point}")
            
            st.markdown("---")


def main():
    """Main application."""
    # Load data
    df = load_data()
    
    if df is None:
        st.error("No data found. Please run the analysis pipeline first.")
        st.markdown("""
        Run the following command to generate data:
        ```bash
        python main.py
        ```
        """)
        return
    
    # Initialize summarizer
    summarizer = get_summarizer(df)
    
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Render selected page
    if page == "Project Report":
        render_project_report(df)
    else:
        render_chatbot(df, summarizer)


if __name__ == "__main__":
    main()
