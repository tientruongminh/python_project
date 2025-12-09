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
        ["Bao Cao Du An", "Phan Tich Khia Canh", "Danh Gia Mo Hinh"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Thong Tin")
    st.sidebar.markdown(
        "He thong phan tich danh gia san pham Walmart "
        "su dung Embeddings, Clustering va LLM (Gemini)."
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
# CHATBOT PAGE - EMBEDDING-BASED ASPECT ANALYSIS
# ============================================================

def render_chatbot(df, summarizer):
    """Render giao diện phân tích khía cạnh với Embeddings."""
    st.markdown('<div class="main-header">Phân Tích Khía Cạnh Đánh Giá Sản Phẩm</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Hệ thống sử dụng **Sentence Embeddings + Clustering** để phân tích khía cạnh từ các đánh giá.
    Không sử dụng rule-based, hoàn toàn dựa trên ngữ nghĩa (semantic).
    
    **Quy trình:**
    1. Lọc reviews theo sản phẩm/danh mục
    2. Tạo embeddings bằng Sentence Transformers
    3. Giảm chiều bằng UMAP
    4. Gom cụm bằng KMeans
    5. Đặt tên và tóm tắt bằng Gemini LLM
    """)
    
    st.markdown("---")
    
    # Chọn mode
    mode = st.radio(
        "Chọn chế độ phân tích:",
        ["Case 1: Phát hiện N khía cạnh phổ biến", "Case 2: Phân tích theo tên khía cạnh"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if mode == "Case 1: Phát hiện N khía cạnh phổ biến":
        render_case1_n_aspects(df, summarizer)
    else:
        render_case2_aspect_name(df, summarizer)


def render_case1_n_aspects(df, summarizer):
    """Case 1: Nhập sản phẩm/danh mục + số khía cạnh."""
    st.markdown("### Case 1: Phát Hiện N Khía Cạnh Phổ Biến")
    
    st.markdown("""
    **Workflow:**
    1. Lọc reviews theo sản phẩm hoặc danh mục
    2. Embedding tất cả reviews
    3. Giảm chiều (UMAP) → Gom cụm (KMeans với k = N)
    4. Đặt tên cho từng cluster bằng LLM
    5. Tóm tắt từng khía cạnh
    """)
    
    # Input fields
    col1, col2 = st.columns(2)
    
    with col1:
        # Chọn danh mục
        categories = ['Tất cả']
        if 'product_category' in df.columns:
            cats = df['product_category'].dropna().unique().tolist()
            cats = [c for c in cats if c not in ['Unknown', 'Other']]
            categories += sorted(cats)
        
        selected_category = st.selectbox("Chọn danh mục sản phẩm:", categories)
        
        # Hoặc nhập tên sản phẩm
        product_name = st.text_input(
            "Hoặc nhập tên sản phẩm:",
            placeholder="Ví dụ: headphones, TV, tablet..."
        )
    
    with col2:
        n_aspects = st.number_input(
            "Số khía cạnh muốn phát hiện:",
            min_value=2,
            max_value=10,
            value=3,
            help="Số cụm (clusters) sẽ được tạo"
        )
        
        max_reviews = st.number_input(
            "Số reviews tối đa:",
            min_value=50,
            max_value=1000,
            value=300,
            help="Giới hạn số reviews để xử lý nhanh hơn"
        )
    
    if st.button("Phân Tích Khía Cạnh", type="primary", key="case1_btn"):
        category = None if selected_category == "Tất cả" else selected_category
        product = product_name if product_name.strip() else None
        
        if not category and not product:
            st.warning("Vui lòng chọn danh mục hoặc nhập tên sản phẩm.")
            return
        
        with st.spinner("Đang phân tích... (có thể mất 1-2 phút)"):
            try:
                # Import EmbeddingAspectSummarizer trực tiếp
                from src.analysis.aspect_summarizer import EmbeddingAspectSummarizer
                
                # Kiểm tra xem có thể sử dụng embedding không
                embedding_summarizer = EmbeddingAspectSummarizer(df)
                
                result = embedding_summarizer.analyze_by_num_aspects(
                    n_aspects=int(n_aspects),
                    product=product,
                    category=category,
                    max_reviews=int(max_reviews)
                )
                
                display_case1_result(result)
                
            except Exception as e:
                st.error(f"Lỗi phân tích: {str(e)}")
                st.info("Đảm bảo đã cài đặt: pip install sentence-transformers umap-learn scikit-learn")


def display_case1_result(result):
    """Hiển thị kết quả Case 1."""
    if not result.get('success'):
        st.error(result.get('error', 'Có lỗi xảy ra'))
        return
    
    st.markdown("---")
    st.markdown(f"### Kết Quả: {result.get('n_aspects')} Khía Cạnh")
    
    # Thông tin tổng quan
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tổng Reviews Phân Tích", f"{result.get('total_reviews', 0):,}")
    with col2:
        st.metric("Số Khía Cạnh", result.get('n_aspects', 0))
    with col3:
        if result.get('category'):
            st.metric("Danh Mục", result.get('category', 'N/A'))
        elif result.get('product'):
            st.metric("Sản Phẩm", result.get('product', 'N/A'))
    
    st.markdown("---")
    
    # Hiển thị từng khía cạnh
    for aspect in result.get('aspects', []):
        with st.container():
            st.markdown(f"#### Khía cạnh {aspect['aspect_id']}: {aspect['aspect_name']}")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.metric("Số Reviews", aspect['review_count'])
            
            with col2:
                st.markdown(f"**Tóm tắt:**")
                st.markdown(aspect['summary'])
            
            # Sample reviews
            if aspect.get('sample_reviews'):
                with st.expander("Xem các đánh giá mẫu"):
                    for i, review in enumerate(aspect['sample_reviews'][:5], 1):
                        truncated = review[:300] + "..." if len(review) > 300 else review
                        st.markdown(f"**{i}.** {truncated}")
            
            st.markdown("---")


def render_case2_aspect_name(df, summarizer):
    """Case 2: Nhập sản phẩm + tên khía cạnh."""
    st.markdown("### Case 2: Phân Tích Theo Tên Khía Cạnh")
    
    st.markdown("""
    **Workflow:**
    1. Lọc reviews theo sản phẩm hoặc danh mục
    2. Embedding reviews + embedding tên khía cạnh
    3. Tính cosine similarity giữa reviews và khía cạnh
    4. Lọc reviews có similarity cao
    5. Tóm tắt các reviews đó bằng LLM
    """)
    
    # Input fields
    col1, col2 = st.columns(2)
    
    with col1:
        # Chọn danh mục
        categories = ['Tất cả']
        if 'product_category' in df.columns:
            cats = df['product_category'].dropna().unique().tolist()
            cats = [c for c in cats if c not in ['Unknown', 'Other']]
            categories += sorted(cats)
        
        selected_category = st.selectbox(
            "Chọn danh mục sản phẩm:", 
            categories,
            key="case2_category"
        )
        
        # Hoặc nhập tên sản phẩm
        product_name = st.text_input(
            "Hoặc nhập tên sản phẩm:",
            placeholder="Ví dụ: headphones, TV, tablet...",
            key="case2_product"
        )
    
    with col2:
        aspect_name = st.text_input(
            "Nhập tên khía cạnh muốn phân tích:",
            placeholder="Ví dụ: sound quality, battery life, shipping speed...",
            help="Hệ thống sẽ tìm các reviews có ngữ nghĩa tương tự"
        )
        
        similarity_threshold = st.slider(
            "Ngưỡng similarity tối thiểu:",
            min_value=0.1,
            max_value=0.8,
            value=0.3,
            step=0.05,
            help="Chỉ lấy reviews có similarity >= ngưỡng này"
        )
    
    if st.button("Phân Tích Khía Cạnh", type="primary", key="case2_btn"):
        if not aspect_name.strip():
            st.warning("Vui lòng nhập tên khía cạnh.")
            return
        
        category = None if selected_category == "Tất cả" else selected_category
        product = product_name if product_name.strip() else None
        
        with st.spinner("Đang phân tích... (có thể mất 1-2 phút)"):
            try:
                from src.analysis.aspect_summarizer import EmbeddingAspectSummarizer
                
                embedding_summarizer = EmbeddingAspectSummarizer(df)
                
                result = embedding_summarizer.analyze_by_aspect_name(
                    aspect_name=aspect_name,
                    product=product,
                    category=category,
                    similarity_threshold=similarity_threshold
                )
                
                display_case2_result(result)
                
            except Exception as e:
                st.error(f"Lỗi phân tích: {str(e)}")
                st.info("Đảm bảo đã cài đặt: pip install sentence-transformers scikit-learn")


def display_case2_result(result):
    """Hiển thị kết quả Case 2."""
    if not result.get('success'):
        st.error(result.get('error', 'Có lỗi xảy ra'))
        return
    
    st.markdown("---")
    st.markdown(f"### Kết Quả: Khía Cạnh \"{result.get('aspect_name')}\"")
    
    # Thông tin tổng quan
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tổng Reviews", f"{result.get('total_reviews_analyzed', 0):,}")
    with col2:
        st.metric("Reviews Liên Quan", f"{result.get('relevant_reviews_count', 0):,}")
    with col3:
        st.metric("Similarity TB", f"{result.get('avg_similarity', 0):.3f}")
    with col4:
        sentiment = result.get('sentiment', {})
        if sentiment.get('positive_pct', 0) > 50:
            st.metric("Sentiment", f"Positive ({sentiment.get('positive_pct')}%)")
        elif sentiment.get('negative_pct', 0) > 30:
            st.metric("Sentiment", f"Negative ({sentiment.get('negative_pct')}%)")
        else:
            st.metric("Sentiment", f"Neutral ({sentiment.get('neutral_pct')}%)")
    
    st.markdown("---")
    
    # Tóm tắt
    st.markdown("#### Tóm Tắt")
    st.markdown(result.get('summary', 'Không có tóm tắt'))
    
    # Sentiment breakdown
    st.markdown("---")
    st.markdown("#### Phân Bổ Sentiment")
    
    sentiment = result.get('sentiment', {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Positive", f"{sentiment.get('positive_pct', 0)}%")
    with col2:
        st.metric("Neutral", f"{sentiment.get('neutral_pct', 0)}%")
    with col3:
        st.metric("Negative", f"{sentiment.get('negative_pct', 0)}%")
    
    # Sample reviews với similarity
    st.markdown("---")
    st.markdown("#### Các Đánh Giá Liên Quan Nhất")
    
    sample_reviews = result.get('sample_reviews', [])
    if sample_reviews:
        for i, item in enumerate(sample_reviews, 1):
            review = item.get('review', '')
            similarity = item.get('similarity', 0)
            
            truncated = review[:400] + "..." if len(review) > 400 else review
            
            with st.container():
                st.markdown(f"**{i}. (Similarity: {similarity:.3f})**")
                st.markdown(f"> {truncated}")
                st.markdown("")
    else:
        st.info("Không tìm thấy reviews liên quan.")


# ============================================================
# MODEL EVALUATION PAGE
# ============================================================

def render_evaluation(df, summarizer):
    """Render model evaluation page."""
    st.markdown('<div class="main-header">Đánh Giá Chất Lượng Mô Hình</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Trang này cho phép đánh giá chất lượng của mô hình Aspect-Based Summarization.
    
    **Các nhóm metrics:**
    1. **Clustering Quality**: Silhouette Score, Calinski-Harabasz, Davies-Bouldin
    2. **Topic Coherence**: Semantic similarity trong từng cluster
    3. **Coverage**: Tỷ lệ reviews được gán aspect
    4. **Summary Quality**: Relevance của summary với source reviews
    """)
    
    st.markdown("---")
    
    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        categories = ['Electronics - Headphones', 'Electronics - TV']
        if 'product_category' in df.columns:
            cats = df['product_category'].dropna().unique().tolist()
            cats = [c for c in cats if c not in ['Unknown', 'Other'] and len(df[df['product_category'] == c]) >= 50]
            if cats:
                categories = sorted(cats)[:20]  # Top 20 categories
        
        selected_category = st.selectbox(
            "Chọn danh mục để đánh giá:", 
            categories,
            key="eval_category"
        )
    
    with col2:
        n_aspects = st.number_input(
            "Số khía cạnh (clusters):",
            min_value=2,
            max_value=10,
            value=3,
            key="eval_n_aspects"
        )
    
    if st.button("Chạy Đánh Giá", type="primary", key="run_eval"):
        with st.spinner("Đang đánh giá mô hình... (có thể mất 1-2 phút)"):
            try:
                from src.analysis.evaluator import AspectModelEvaluator
                from src.analysis.aspect_summarizer import EmbeddingAspectSummarizer
                
                # Create summarizer
                emb_summarizer = EmbeddingAspectSummarizer(df, fast_mode=True, use_cache=True)
                
                # Run analysis
                result = emb_summarizer.analyze_by_num_aspects(
                    n_aspects=int(n_aspects),
                    category=selected_category,
                    max_reviews=200
                )
                
                if not result.get('success'):
                    st.error(result.get('error', 'Phân tích thất bại'))
                    return
                
                # Get data for evaluation
                filtered_df = emb_summarizer._get_reviews_for_product(category=selected_category, max_reviews=200)
                reviews = filtered_df['review'].tolist()
                
                if len(reviews) < n_aspects:
                    st.error(f"Không đủ reviews ({len(reviews)})")
                    return
                
                embeddings = emb_summarizer._create_embeddings(reviews)
                reduced = emb_summarizer._reduce_dimensions(embeddings)
                labels = emb_summarizer._cluster_embeddings(reduced, int(n_aspects))
                
                # Build clusters and summaries dicts
                clusters = {}
                summaries = {}
                for i, label in enumerate(labels):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(reviews[i])
                
                for aspect in result['aspects']:
                    cluster_id = aspect['aspect_id'] - 1
                    summaries[cluster_id] = aspect['summary']
                
                # Run evaluation
                evaluator = AspectModelEvaluator(emb_summarizer.embedding_model)
                evaluation = evaluator.run_full_evaluation(
                    embeddings=reduced,
                    labels=labels,
                    clusters=clusters,
                    summaries=summaries,
                    total_reviews=len(reviews)
                )
                
                # Display results
                display_evaluation_results(evaluation, result)
                
            except ImportError as e:
                st.error(f"Thiếu dependencies: {e}")
                st.info("Cài đặt: pip install sentence-transformers scikit-learn")
            except Exception as e:
                st.error(f"Lỗi: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def display_evaluation_results(evaluation: dict, analysis_result: dict):
    """Display evaluation results."""
    st.markdown("---")
    st.markdown("## Kết Quả Đánh Giá")
    
    # Overall Score
    overall = evaluation.get('overall_assessment', {})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        score = overall.get('overall_score', 0)
        st.metric("Overall Score", f"{score:.1%}")
    with col2:
        st.metric("Grade", overall.get('grade', 'N/A'))
    with col3:
        st.metric("Tổng Reviews", evaluation.get('coverage', {}).get('total_reviews', 0))
    
    st.markdown("---")
    
    # Component scores
    st.markdown("### Điểm Theo Thành Phần")
    
    components = overall.get('components', {})
    comp_df = pd.DataFrame([
        {'Thành phần': k.capitalize(), 'Điểm': f"{v:.1%}"}
        for k, v in components.items()
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Detailed metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 1. Clustering Quality")
        cluster = evaluation.get('clustering_quality', {})
        st.markdown(f"""
        - **Silhouette Score**: {cluster.get('silhouette_score', 0):.4f}
          - {cluster.get('silhouette_interpretation', 'N/A')}
        - **Calinski-Harabasz**: {cluster.get('calinski_harabasz', 0):.2f}
        - **Davies-Bouldin**: {cluster.get('davies_bouldin', 0):.4f}
          - {cluster.get('davies_interpretation', 'N/A')}
        """)
        
        st.markdown("### 2. Topic Coherence")
        coherence = evaluation.get('topic_coherence', {})
        st.markdown(f"""
        - **Overall Coherence**: {coherence.get('overall_coherence', 0):.4f}
        - **Interpretation**: {coherence.get('interpretation', 'N/A')}
        """)
    
    with col2:
        st.markdown("### 3. Coverage")
        coverage = evaluation.get('coverage', {})
        st.markdown(f"""
        - **Tổng reviews**: {coverage.get('total_reviews', 0):,}
        - **Reviews có aspect**: {coverage.get('reviews_with_aspects', 0):,}
        - **Coverage Rate**: {coverage.get('coverage_rate', 0):.1f}%
        - **Interpretation**: {coverage.get('coverage_interpretation', 'N/A')}
        """)
        
        if coverage.get('cluster_balance'):
            balance = coverage['cluster_balance']
            st.markdown(f"""
            **Cluster Balance:**
            - Min: {balance.get('min_size', 0)}, Max: {balance.get('max_size', 0)}
            - CV: {balance.get('coefficient_of_variation', 0):.3f}
            - {balance.get('balance_interpretation', 'N/A')}
            """)
        
        st.markdown("### 4. Summary Quality")
        summary = evaluation.get('summary_quality', {})
        st.markdown(f"""
        - **Avg Relevance**: {summary.get('avg_relevance', 0):.4f}
        - **Summaries evaluated**: {summary.get('n_summaries_evaluated', 0)}
        """)
    
    st.markdown("---")
    
    # Interpretation
    st.markdown("### Giải Thích Metrics")
    
    with st.expander("Xem chi tiết về các metrics"):
        st.markdown("""
        #### Clustering Quality
        
        | Metric | Ý nghĩa | Giá trị tốt |
        |--------|---------|-------------|
        | Silhouette Score | Độ tách biệt giữa clusters | ≥ 0.5 |
        | Calinski-Harabasz | Tỷ lệ variance between/within | Càng cao càng tốt |
        | Davies-Bouldin | Độ overlap giữa clusters | ≤ 1.0 |
        
        #### Topic Coherence
        
        Đo lường độ mạch lạc của các topics/aspects được phát hiện.
        - **> 0.6**: Rất tốt - Topics rõ ràng, mạch lạc
        - **0.4-0.6**: Tốt - Topics có ý nghĩa
        - **0.25-0.4**: Trung bình - Có overlap
        - **< 0.25**: Yếu - Topics không rõ ràng
        
        #### Coverage
        
        Tỷ lệ reviews được gán vào ít nhất 1 aspect.
        - **> 70%**: Tốt
        - **40-70%**: Trung bình
        - **< 40%**: Yếu
        
        #### Summary Quality
        
        Độ liên quan giữa summary và source reviews.
        - **> 0.7**: Rất tốt
        - **0.5-0.7**: Tốt
        - **0.3-0.5**: Trung bình
        - **< 0.3**: Yếu
        """)
    
    # Aspects found
    st.markdown("---")
    st.markdown("### Các Khía Cạnh Được Phát Hiện")
    
    for aspect in analysis_result.get('aspects', []):
        with st.expander(f"Khía cạnh {aspect['aspect_id']}: {aspect['aspect_name']} ({aspect['review_count']} reviews)"):
            st.markdown(f"**Tóm tắt:** {aspect['summary']}")
            st.markdown("**Sample reviews:**")
            for i, rev in enumerate(aspect.get('sample_reviews', [])[:3], 1):
                st.markdown(f"{i}. {rev[:200]}...")


def main():
    """Main application."""
    df = load_data()
    
    if df is None:
        st.error("Không tìm thấy dữ liệu. Hãy chạy pipeline trước.")
        st.code("python main.py", language="bash")
        return
    
    summarizer = get_summarizer(df)
    page = render_sidebar()
    
    if page == "Bao Cao Du An":
        render_project_report(df)
    elif page == "Phan Tich Khia Canh":
        render_chatbot(df, summarizer)
    elif page == "Danh Gia Mo Hinh":
        render_evaluation(df, summarizer)


if __name__ == "__main__":
    main()
