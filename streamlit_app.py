"""
Streamlit Dashboard for Walmart Product Review Analysis.

Features:
- Comprehensive Project Report
- Gemini-powered Chatbot for natural language Q&A
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

# Custom CSS
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
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #1E88E5;
        padding-bottom: 0.3rem;
    }
    .subsection-header {
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1rem;
        color: #424242;
    }
    .chat-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1E88E5;
    }
    .bot-message {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .methodology-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load processed data."""
    data_path = OUTPUT_DIR / "sentiment_analysis.csv"
    if data_path.exists():
        return pd.read_csv(data_path)
    
    data_path = settings.processed_data_path
    if data_path.exists():
        return pd.read_csv(data_path)
        
    return None


@st.cache_resource
def get_summarizer(df):
    """Get or create AspectSummarizer."""
    return AspectSummarizer(df)


@st.cache_resource
def get_gemini_client():
    """Initialize Gemini client for chatbot."""
    try:
        from src.clustering.gemini_client import GeminiClient
        client = GeminiClient()
        if client.is_available:
            return client
    except Exception as e:
        st.warning(f"Gemini not available: {e}")
    return None


def render_sidebar():
    """Render sidebar with navigation."""
    st.sidebar.markdown("## Menu")
    
    page = st.sidebar.radio(
        "Select Page",
        ["Bao Cao Du An", "Chatbot"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Thong Tin")
    st.sidebar.markdown(
        "He thong phan tich danh gia san pham Walmart "
        "su dung BERTopic va LLM (Gemini)."
    )
    
    return page


# ============================================================
# PROJECT REPORT PAGE
# ============================================================

def render_project_report(df):
    """Render comprehensive project report."""
    st.markdown('<div class="main-header">Bao Cao Chi Tiet Du An Phan Tich Danh Gia San Pham Walmart</div>', unsafe_allow_html=True)
    
    # Table of Contents
    st.markdown("""
    **Muc Luc:**
    1. [Tong Quan Du An](#1-tong-quan-du-an)
    2. [Du Lieu va Tien Xu Ly](#2-du-lieu-va-tien-xu-ly)
    3. [Phan Tich Kham Pha Du Lieu (EDA)](#3-phan-tich-kham-pha-du-lieu-eda)
    4. [Mo Hinh va Phuong Phap](#4-mo-hinh-va-phuong-phap)
    5. [Ket Qua Phan Tich](#5-ket-qua-phan-tich)
    6. [Ket Luan va Khuyen Nghi](#6-ket-luan-va-khuyen-nghi)
    """)
    
    st.markdown("---")
    
    # Section 1: Project Overview
    render_section1_overview(df)
    
    # Section 2: Data & Preprocessing
    render_section2_preprocessing(df)
    
    # Section 3: EDA
    render_section3_eda(df)
    
    # Section 4: Models & Methods
    render_section4_models()
    
    # Section 5: Results
    render_section5_results(df)
    
    # Section 6: Conclusions
    render_section6_conclusions(df)


def render_section1_overview(df):
    """Section 1: Project Overview."""
    st.markdown('<div class="section-header">1. Tong Quan Du An</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Muc tieu:** Xay dung pipeline phan tich danh gia san pham tu Walmart de:
    - Hieu nguoi dung noi gi ve san pham
    - Phat hien cac khia canh (aspects) duoc de cap nhieu nhat
    - Phan tich cam xuc (sentiment) theo tung khia canh
    - Tao chatbot tra loi cau hoi bang ngon ngu tu nhien
    
    **Cong nghe su dung:**
    - Python 3.12, Pandas, Plotly
    - BERTopic cho topic modeling
    - Google Gemini API cho LLM
    - Streamlit cho dashboard
    """)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tong So Danh Gia", f"{len(df):,}")
    
    with col2:
        unique_products = df['pageurl'].nunique() if 'pageurl' in df.columns else 0
        st.metric("So San Pham", f"{unique_products:,}")
    
    with col3:
        categories = df['product_category'].nunique() if 'product_category' in df.columns else 0
        st.metric("So Danh Muc", f"{categories:,}")
    
    with col4:
        avg_rating = df['rating'].mean() if 'rating' in df.columns else 0
        st.metric("Diem Trung Binh", f"{avg_rating:.2f}/5")


def render_section2_preprocessing(df):
    """Section 2: Data & Preprocessing."""
    st.markdown('<div class="section-header">2. Du Lieu va Tien Xu Ly</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="subsection-header">2.1 Nguon Du Lieu</div>', unsafe_allow_html=True)
    st.markdown("""
    - **Nguon:** Kaggle - Walmart Product Reviews Dataset
    - **Thoi gian:** 01/04/2020 - 30/06/2020 (da shift lui 10 nam)
    - **So luong ban dau:** 29,997 dong
    - **So cot:** 18 cot goc
    """)
    
    st.markdown('<div class="subsection-header">2.2 Quy Trinh Tien Xu Ly (5 Chieu Chat Luong)</div>', unsafe_allow_html=True)
    
    preprocessing_data = {
        'Chieu Chat Luong': ['1. Completeness', '2. Accuracy', '3. Validity', '4. Timeliness', '5. Uniqueness'],
        'Mo Ta': [
            'Xu ly missing values: dien title tu duplicate URLs, dien reviewer name',
            'Sua gia tri sai: negative votes -> 0, rating ngoai [1,5]',
            'Chuan hoa format: verified_purchaser -> Yes/No/Unknown',
            'Shift dates lui 10 nam, validate date ranges',
            'Xoa duplicate: 355 dong trung lap (1.18%)'
        ],
        'Ket Qua': [
            '10,345 titles da dien, 1,620 names',
            'Clipped invalid ratings to [1,5]',
            '100% consistent format',
            '29,997 dates shifted',
            f'{len(df):,} dong sau xu ly'
        ]
    }
    st.dataframe(pd.DataFrame(preprocessing_data), use_container_width=True, hide_index=True)
    
    st.markdown('<div class="subsection-header">2.3 Cac Cot Moi Duoc Tao</div>', unsafe_allow_html=True)
    
    new_cols = {
        'Cot Moi': ['product_id', 'total_votes', 'helpfulness_score', 'word_count', 
                   'rating_sentiment', 'review_year', 'review_month', 'review_year_month'],
        'Cong Thuc/Nguon': [
            'Extract tu PageURL',
            'review_upvotes + review_downvotes',
            'Wilson Score Formula',
            'len(review.split())',
            'Positive (4-5), Neutral (3), Negative (1-2)',
            'Extract tu review_date',
            'Extract tu review_date',
            'YYYY-MM format'
        ]
    }
    st.dataframe(pd.DataFrame(new_cols), use_container_width=True, hide_index=True)


def render_section3_eda(df):
    """Section 3: EDA."""
    st.markdown('<div class="section-header">3. Phan Tich Kham Pha Du Lieu (EDA)</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subsection-header">3.1 Phan Bo Rating</div>', unsafe_allow_html=True)
        if 'rating' in df.columns:
            rating_counts = df['rating'].value_counts().sort_index()
            fig = px.bar(
                x=rating_counts.index,
                y=rating_counts.values,
                labels={'x': 'Rating', 'y': 'So Luong'},
                color=rating_counts.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Insights
            mode_rating = rating_counts.idxmax()
            st.markdown(f"""
            **Nhan xet:**
            - Rating pho bien nhat: **{mode_rating} sao** ({rating_counts.max():,} danh gia)
            - Ty le 5 sao: **{rating_counts.get(5, 0)/len(df)*100:.1f}%**
            - Ty le 1 sao: **{rating_counts.get(1, 0)/len(df)*100:.1f}%**
            """)
    
    with col2:
        st.markdown('<div class="subsection-header">3.2 Phan Bo Sentiment</div>', unsafe_allow_html=True)
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
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            pos_pct = sentiment_counts.get('Positive', 0) / len(df) * 100
            neg_pct = sentiment_counts.get('Negative', 0) / len(df) * 100
            st.markdown(f"""
            **Nhan xet:**
            - Positive: **{pos_pct:.1f}%**
            - Negative: **{neg_pct:.1f}%**
            - Ratio Positive/Negative: **{pos_pct/max(neg_pct,1):.1f}x**
            """)
    
    # Category Analysis
    st.markdown('<div class="subsection-header">3.3 Phan Tich Theo Danh Muc</div>', unsafe_allow_html=True)
    if 'product_category' in df.columns:
        category_stats = df.groupby('product_category').agg({
            'rating': ['count', 'mean', 'std']
        }).round(2)
        category_stats.columns = ['So Danh Gia', 'Rating TB', 'Do Lech Chuan']
        category_stats = category_stats.sort_values('So Danh Gia', ascending=False).head(10)
        
        fig = px.bar(
            category_stats.reset_index(),
            x='product_category',
            y='So Danh Gia',
            color='Rating TB',
            color_continuous_scale='RdYlGn',
            labels={'product_category': 'Danh Muc'}
        )
        fig.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        **Nhan xet:**
        - Danh muc nhieu danh gia nhat: **{category_stats.index[0]}** ({category_stats.iloc[0]['So Danh Gia']:,.0f} danh gia)
        - Danh muc co rating cao nhat trong top 10: **{category_stats['Rating TB'].idxmax()}** ({category_stats['Rating TB'].max():.2f})
        """)
    
    # Time Trend
    st.markdown('<div class="subsection-header">3.4 Xu Huong Theo Thoi Gian</div>', unsafe_allow_html=True)
    if 'review_year_month' in df.columns:
        trend = df.groupby('review_year_month').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        trend.columns = ['Thang', 'So Danh Gia', 'Rating TB']
        trend = trend.sort_values('Thang').tail(24)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=trend['Thang'], y=trend['So Danh Gia'], name='So Danh Gia', yaxis='y'))
        fig.add_trace(go.Scatter(x=trend['Thang'], y=trend['Rating TB'], name='Rating TB', yaxis='y2', line=dict(color='red', width=2)))
        
        fig.update_layout(
            yaxis=dict(title='So Danh Gia'),
            yaxis2=dict(title='Rating TB', overlaying='y', side='right', range=[1, 5]),
            height=400,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)


def render_section4_models():
    """Section 4: Models & Methods."""
    st.markdown('<div class="section-header">4. Mo Hinh va Phuong Phap</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="subsection-header">4.1 Product Clustering (Gemini API)</div>', unsafe_allow_html=True)
    st.markdown("""
    **Muc tieu:** Nhom san pham vao cac danh muc tu dong
    
    **Phuong phap:**
    1. Lay ten san pham (title) tu data
    2. Gui batch 20 san pham den Gemini API
    3. Gemini phan loai vao cac category
    4. Consolidate cac category nho (<10 san pham) vao "Other"
    
    **Tai sao chon Gemini:**
    - Hieu ngon ngu tu nhien tot
    - Khong can training data
    - Co the phan loai san pham moi ma chua thay bao gio
    
    **Ket qua:** 1,258 san pham -> 108 danh muc
    """)
    
    st.markdown('<div class="subsection-header">4.2 Aspect Extraction</div>', unsafe_allow_html=True)
    st.markdown("""
    **Muc tieu:** Phat hien cac khia canh (aspects) trong reviews
    
    **Phuong phap:**
    1. **Keyword-based:** Dinh nghia keywords cho moi aspect
       - quality: "quality", "well made", "durable"
       - price: "price", "expensive", "cheap", "value"
       - shipping: "shipping", "delivery", "arrived"
    
    2. **Context-aware sentiment:** Phan tich sentiment quanh keyword
       - Window size: 5 tu truoc/sau
       - Positive keywords: "great", "excellent", "love"
       - Negative keywords: "bad", "terrible", "broken"
    
    **Ket qua:** 10 aspects chinh duoc extract
    """)
    
    st.markdown('<div class="subsection-header">4.3 LLM Summarization (Gemini)</div>', unsafe_allow_html=True)
    st.markdown("""
    **Muc tieu:** Tom tat noi dung nguoi dung noi ve moi aspect
    
    **Phuong phap:**
    1. Lay sample reviews de cap den aspect
    2. Tao prompt cho Gemini:
       ```
       Analyze these reviews about "{aspect}".
       Provide: overall sentiment, common praises/complaints, key takeaways.
       ```
    3. Gemini tra ve summary bang ngon ngu tu nhien
    
    **Tai sao can LLM:**
    - Keyword-based chi cho biet "co de cap"
    - LLM cho biet "nguoi dung noi GI" cu the
    - Summary de hieu hon reading 1000 reviews
    
    **Fallback:** Neu API fail, dung keyword-based summary
    """)


def render_section5_results(df):
    """Section 5: Results."""
    st.markdown('<div class="section-header">5. Ket Qua Phan Tich</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="subsection-header">5.1 Top Danh Muc San Pham</div>', unsafe_allow_html=True)
    if 'product_category' in df.columns:
        top_cats = df.groupby('product_category').agg({
            'rating': ['count', 'mean'],
            'rating_sentiment': lambda x: (x == 'Positive').mean() * 100
        }).round(2)
        top_cats.columns = ['So Danh Gia', 'Rating TB', '% Positive']
        top_cats = top_cats.sort_values('So Danh Gia', ascending=False).head(10)
        st.dataframe(top_cats, use_container_width=True)
    
    st.markdown('<div class="subsection-header">5.2 Phan Tich Aspects</div>', unsafe_allow_html=True)
    
    aspect_cols = [col for col in df.columns if col.startswith('has_')]
    if aspect_cols:
        aspect_data = []
        for col in aspect_cols:
            aspect = col.replace('has_', '')
            mentions = df[col].sum() if df[col].dtype == bool else (df[col] == True).sum()
            sent_col = f'{aspect}_sentiment'
            if sent_col in df.columns:
                pos_rate = (df.loc[df[col] == True, sent_col] == 'positive').mean() * 100 if mentions > 0 else 0
                neg_rate = (df.loc[df[col] == True, sent_col] == 'negative').mean() * 100 if mentions > 0 else 0
            else:
                pos_rate = neg_rate = 0
            
            aspect_data.append({
                'Aspect': aspect,
                'So De Cap': int(mentions),
                'Ty Le De Cap': f"{mentions/len(df)*100:.1f}%",
                '% Positive': f"{pos_rate:.1f}%",
                '% Negative': f"{neg_rate:.1f}%"
            })
        
        aspect_df = pd.DataFrame(aspect_data).sort_values('So De Cap', ascending=False)
        st.dataframe(aspect_df, use_container_width=True, hide_index=True)
        
        # Visualization
        fig = px.bar(
            aspect_df.head(10),
            x='Aspect',
            y='So De Cap',
            color='Aspect',
            title='Top 10 Aspects Duoc De Cap'
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)


def render_section6_conclusions(df):
    """Section 6: Conclusions."""
    st.markdown('<div class="section-header">6. Ket Luan va Khuyen Nghi</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="subsection-header">6.1 Ket Luan Chinh</div>', unsafe_allow_html=True)
    
    # Calculate key insights
    if 'rating' in df.columns and 'product_category' in df.columns:
        avg_rating = df['rating'].mean()
        best_cat = df.groupby('product_category')['rating'].mean().idxmax()
        worst_cat = df.groupby('product_category')['rating'].mean().idxmin()
        
        st.markdown(f"""
        1. **Chat luong du lieu:** Sau xu ly, 27/29 cot co 0% missing values
        2. **Danh gia tich cuc:** Phan lon danh gia la tich cuc (rating TB: {avg_rating:.2f})
        3. **Danh muc tot nhat:** {best_cat}
        4. **Can cai thien:** {worst_cat}
        """)
    
    st.markdown('<div class="subsection-header">6.2 Khuyen Nghi Kinh Doanh</div>', unsafe_allow_html=True)
    st.markdown("""
    **[HIGH] Cai thien Quality:**
    - Aspect "quality" co ty le negative cao nhat
    - De xuat: Review lai quy trinh kiem tra chat luong
    
    **[MEDIUM] Tang cuong Customer Service:**
    - Nhieu danh gia negative lien quan den customer_service
    - De xuat: Training nhan vien ho tro
    
    **[LOW] Toi uu Shipping:**
    - Shipping duoc de cap nhieu nhung phan lon neutral
    - De xuat: Cai thien thoi gian giao hang
    """)
    
    st.markdown('<div class="subsection-header">6.3 Huong Phat Trien</div>', unsafe_allow_html=True)
    st.markdown("""
    1. **Tich hop BERTopic:** Auto-discover aspects thay vi keyword-based
    2. **Real-time monitoring:** Theo doi sentiment theo thoi gian thuc
    3. **Multi-language:** Mo rong sang tieng Viet va cac ngon ngu khac
    4. **API integration:** Xay dung API cho cac he thong khac su dung
    """)


# ============================================================
# CHATBOT PAGE
# ============================================================

def render_chatbot(df, summarizer):
    """Render Gemini-powered chatbot."""
    st.markdown('<div class="main-header">Chatbot Phan Tich Danh Gia</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Chatbot su dung Gemini AI de tra loi cau hoi ve danh gia san pham.
    
    **Cac cau hoi mau:**
    - "Nguoi dung nói gi ve chat luong san pham?"
    - "3 khia canh pho bien nhat cua Electronics - Headphones la gi?"
    - "Sound quality duoc danh gia nhu the nao?"
    """)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        role_class = "user-message" if message["role"] == "user" else "bot-message"
        st.markdown(f'<div class="{role_class}">{message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Nhap cau hoi cua ban...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.markdown(f'<div class="user-message">{user_input}</div>', unsafe_allow_html=True)
        
        # Generate response
        with st.spinner("Dang phan tich..."):
            response = generate_gemini_response(user_input, df, summarizer)
        
        # Add bot response
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(f'<div class="bot-message">{response}</div>', unsafe_allow_html=True)
        
        st.rerun()
    
    # Clear chat button
    if st.button("Xoa lich su chat"):
        st.session_state.messages = []
        st.rerun()


def generate_gemini_response(query: str, df: pd.DataFrame, summarizer) -> str:
    """Generate response using Gemini API."""
    
    # Get Gemini client
    gemini_client = get_gemini_client()
    
    if gemini_client is None:
        return generate_fallback_response(query, df, summarizer)
    
    # Prepare context about the data
    context = prepare_data_context(df)
    
    # Detect query intent and get relevant data
    relevant_data = get_relevant_data_for_query(query, df, summarizer)
    
    # Build prompt
    prompt = f"""Ban la mot chuyen gia phan tich danh gia san pham. Hay tra loi cau hoi sau dua tren du lieu:

**Du lieu tong quan:**
{context}

**Du lieu lien quan den cau hoi:**
{relevant_data}

**Cau hoi cua nguoi dung:**
{query}

**Huong dan tra loi:**
- Tra loi bang tieng Viet
- Su dung ngon ngu tu nhien, de hieu
- Dua ra so lieu cu the neu co
- Neu khong co du lieu, noi ro la khong tim thay
- Tra loi ngan gon nhung day du thong tin

**Tra loi:**"""

    try:
        response = gemini_client.generate(prompt, max_tokens=500)
        return response.strip() if response else generate_fallback_response(query, df, summarizer)
    except Exception as e:
        st.warning(f"Gemini API error: {e}")
        return generate_fallback_response(query, df, summarizer)


def prepare_data_context(df: pd.DataFrame) -> str:
    """Prepare context string about the dataset."""
    avg_rating = df['rating'].mean() if 'rating' in df.columns else None
    avg_rating_str = f"{avg_rating:.2f}" if avg_rating else 'N/A'
    
    context = f"""
- Tong so danh gia: {len(df):,}
- So san pham: {df['pageurl'].nunique() if 'pageurl' in df.columns else 'N/A'}
- So danh muc: {df['product_category'].nunique() if 'product_category' in df.columns else 'N/A'}
- Rating trung binh: {avg_rating_str}
"""
    
    if 'product_category' in df.columns:
        top_cats = df['product_category'].value_counts().head(5)
        context += "\nTop 5 danh muc:\n"
        for cat, count in top_cats.items():
            context += f"- {cat}: {count:,} danh gia\n"
    
    return context


def get_relevant_data_for_query(query: str, df: pd.DataFrame, summarizer) -> str:
    """Extract relevant data based on query."""
    query_lower = query.lower()
    
    # Check for aspect query
    aspects = ['quality', 'price', 'shipping', 'sound', 'battery', 'screen', 'delivery', 
               'value', 'customer service', 'packaging', 'durability', 'chat luong', 
               'gia', 'giao hang', 'am thanh', 'pin', 'man hinh']
    
    found_aspect = None
    for aspect in aspects:
        if aspect in query_lower:
            found_aspect = aspect
            break
    
    if found_aspect:
        # Get aspect summary
        result = summarizer.summarize_aspect(found_aspect, max_reviews=30)
        return f"""
Aspect: {found_aspect}
So luong de cap: {result['review_count']}
Sentiment: {result['sentiment']}
Ty le positive: {result.get('sentiment_scores', {}).get('positive', 0)}%
Ty le negative: {result.get('sentiment_scores', {}).get('negative', 0)}%
Sample reviews: {result.get('sample_reviews', [])[:3]}
"""
    
    # Check for category query
    if 'product_category' in df.columns:
        for cat in df['product_category'].unique():
            if str(cat).lower() in query_lower or query_lower in str(cat).lower():
                cat_df = df[df['product_category'] == cat]
                return f"""
Danh muc: {cat}
So danh gia: {len(cat_df):,}
Rating TB: {cat_df['rating'].mean():.2f}
5 sao: {(cat_df['rating'] == 5).sum():,}
1 sao: {(cat_df['rating'] == 1).sum():,}
"""
    
    # Check for "top aspects" query
    if 'khia canh' in query_lower or 'aspect' in query_lower or 'pho bien' in query_lower:
        # Extract number if present
        import re
        numbers = re.findall(r'\d+', query)
        n = int(numbers[0]) if numbers else 3
        
        # Get top aspects across all data
        aspect_counts = {}
        for col in df.columns:
            if col.startswith('has_'):
                aspect = col.replace('has_', '')
                count = df[col].sum() if df[col].dtype == bool else (df[col] == True).sum()
                aspect_counts[aspect] = count
        
        sorted_aspects = sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True)[:n]
        return f"Top {n} aspects duoc de cap nhieu nhat:\n" + "\n".join([f"- {a}: {c:,} lan" for a, c in sorted_aspects])
    
    # Default: return general stats
    return "Khong tim thay du lieu cu the. Hay thu hoi ve mot aspect hoac danh muc san pham cu the."


def generate_fallback_response(query: str, df: pd.DataFrame, summarizer) -> str:
    """Generate response without Gemini API."""
    query_lower = query.lower()
    
    # Simple keyword matching
    if 'quality' in query_lower or 'chat luong' in query_lower:
        result = summarizer.summarize_aspect('quality', max_reviews=20)
        return f"Ve chat luong: {result['summary']}\nSo luong de cap: {result['review_count']}, Sentiment: {result['sentiment']}"
    
    if 'price' in query_lower or 'gia' in query_lower:
        result = summarizer.summarize_aspect('price', max_reviews=20)
        return f"Ve gia ca: {result['summary']}\nSo luong de cap: {result['review_count']}, Sentiment: {result['sentiment']}"
    
    if 'shipping' in query_lower or 'giao hang' in query_lower:
        result = summarizer.summarize_aspect('shipping', max_reviews=20)
        return f"Ve giao hang: {result['summary']}\nSo luong de cap: {result['review_count']}, Sentiment: {result['sentiment']}"
    
    return f"Toi co the giup ban phan tich cac khia canh nhu: quality, price, shipping, sound, battery, screen. Hay hoi cu the ve mot khia canh nao do!"


def main():
    """Main application."""
    df = load_data()
    
    if df is None:
        st.error("Khong tim thay du lieu. Hay chay pipeline truoc.")
        st.code("python main.py", language="bash")
        return
    
    summarizer = get_summarizer(df)
    page = render_sidebar()
    
    if page == "Bao Cao Du An":
        render_project_report(df)
    else:
        render_chatbot(df, summarizer)


if __name__ == "__main__":
    main()
