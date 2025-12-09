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
        ["Bao Cao Du An", "Phan Tich Khia Canh", "RAG Query", "Danh Gia Mo Hinh"],
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
    st.markdown('<div class="section-header">2. Dữ Liệu và Quy Trinh Xử Lý Chi Tiết</div>', unsafe_allow_html=True)
    
    st.markdown("### 2.1 Dữ Liệu Ban Đầu (Raw Data)")
    st.markdown("""
    Bộ dữ liệu ban đầu bao gồm **29,997 dòng** đánh giá sản phẩm Walmart.
    
    **Các vấn đề chính được phát hiện:**
    1. **Thiếu dữ liệu nghiêm trọng (Completeness):**
       - Cột `title` thiếu 90.9% (27,276 dòng).
       - Cột `reviewer_name` thiếu 5.4% (1,620 dòng).
       - Các cột phân phối sao (`five_star`, `one_star`...) thiếu 0.3%.
    2. **Dữ liệu rác/lỗi (Accuracy):**
       - Một số `rating` nằm ngoài khoảng [1, 5].
       - Số lượng vote tiêu cực (`negative_votes`) có giá trị âm.
    3. **Định dạng không nhất quán (Validity):**
       - Cột `verified_purchaser` chứa nhiều giá trị ("Yes", "yes", "true", "True").
       - Định dạng ngày tháng không đồng nhất.
    4. **Dư thừa (Uniqueness):**
       - 355 dòng bị trùng lặp hoàn toàn.
       - URL chứa tham số tracking thừa.
    """)
    
    st.markdown("### 2.2 Các Bước Xử Lý (Data Cleaning Pipeline)")
    st.success("""
    **Bước 1: Imputation (Điền dữ liệu thiếu)**
    - **Product Title:** Sử dụng chiến lược **Product ID Matching**. Tìm các dòng có cùng Product ID (từ URL) nhưng có Title, sau đó copy Title sang các dòng bị thiếu. Fill được **10,345 titles**. Số còn lại gán "Unknown Product".
    - **Reviewer Name:** Điền giá trị mặc định "Anonymous".
    - **Star Distribution:** Điền giá trị 0 cho các phân phối sao bị thiếu.
    
    **Bước 2: Cleaning & Validation**
    - **Rating:** Clip giá trị về khoảng [1, 5].
    - **Text Cleaning:** Loại bỏ HTML tags, khoảng trắng thừa trong `review` và `title`.
    - **Normalization:** Chuẩn hóa `verified_purchaser` về Yes/No/Unknown.
    - **Duplicate Removal:** Xóa 355 dòng trùng lặp.
    
    **Bước 3: Feature Engineering (Tạo đặc trưng mới)**
    - `helpfulness_score`: Tính theo công thức Wilson Score (cân bằng giữa upvotes và total votes).
    - `sentiment_category`: Phân loại dựa trên rating (4-5: Positive, 3: Neutral, 1-2: Negative).
    - `review_length`: Độ dài đánh giá (số từ).
    """)
    
    st.info(f"**Kết quả sau xử lý:** Bộ dữ liệu sạch gồm **{len(df):,} dòng**, sẵn sàng cho phân tích mô hình.")


def render_section3_eda(df):
    """Section 3: EDA."""
    st.markdown('<div class="section-header">3. Phân Tích Khám Phá Dữ Liệu (EDA)</div>', unsafe_allow_html=True)
    
    st.markdown("Trong phần này, chúng ta sẽ đi sâu vào các đặc điểm thống kê của dữ liệu để hiểu rõ hành vi người dùng.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subsection-header">3.1 Phân Bố Rating (Điểm Đánh Giá)</div>', unsafe_allow_html=True)
        if 'rating' in df.columns:
            rating_counts = df['rating'].value_counts().sort_index()
            fig = px.bar(
                x=rating_counts.index,
                y=rating_counts.values,
                labels={'x': 'Rating', 'y': 'Số Lượng Reviews'},
                color=rating_counts.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Insights
            mode_rating = rating_counts.idxmax()
            st.info(f"""
            **Insight:**
            - Phần lớn đánh giá là **5 sao** ({rating_counts.max():,} reviews), chiếm **{rating_counts.get(5, 0)/len(df)*100:.1f}%**.
            - Điều này cho thấy dữ liệu bị lệch về phía tích cực (Positively Skewed), một đặc điểm chung của E-commerce reviews.
            - Tuy nhiên, vẫn có **{rating_counts.get(1, 0):,}** đánh giá 1 sao cần lưu ý.
            """)
    
    with col2:
        st.markdown('<div class="subsection-header">3.2 Phân Bố Cảm Xúc (Sentiment)</div>', unsafe_allow_html=True)
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
            st.info(f"""
            **Insight:**
            - **{pos_pct:.1f}%** khách hàng hài lòng với sản phẩm.
            - Chỉ có một tỷ lệ nhỏ ({sentiment_counts.get('Negative', 0)/len(df)*100:.1f}%) là tiêu cực.
            - Tỷ lệ này khẳng định lại xu hướng tích cực của tập dữ liệu này.
            """)
    
    # Category Analysis
    st.markdown('<div class="subsection-header">3.3 Hiệu Suất Theo Danh Mục</div>', unsafe_allow_html=True)
    if 'product_category' in df.columns:
        cat_agg = df.groupby('product_category').agg({
            'rating': ['count', 'mean'],
            'helpfulness_score': 'mean'
        }).reset_index()
        cat_agg.columns = ['Category', 'Reviews', 'Avg Rating', 'Avg Helpfulness']
        best_cats = cat_agg[cat_agg['Reviews'] > 50].sort_values('Avg Rating', ascending=False).head(5)
        worst_cats = cat_agg[cat_agg['Reviews'] > 50].sort_values('Avg Rating', ascending=True).head(5)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top 5 Danh Mục Tốt Nhất (Rating cao nhất)**")
            st.table(best_cats[['Category', 'Avg Rating']].set_index('Category'))
            
        with c2:
            st.markdown("**Top 5 Danh Mục Cần Cải Thiện (Rating thấp nhất)**")
            st.table(worst_cats[['Category', 'Avg Rating']].set_index('Category'))
    
    # Time Trend
    st.markdown('<div class="subsection-header">3.4 Xu Hướng Theo Thời Gian</div>', unsafe_allow_html=True)
    if 'review_year_month' in df.columns:
        trend = df.groupby('review_year_month').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        trend.columns = ['Tháng', 'Review Count', 'Avg Rating']
        trend = trend.sort_values('Tháng')
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=trend['Tháng'], y=trend['Review Count'], name='Số Lượng Reviews', marker_color='#90CAF9'))
        fig.add_trace(go.Scatter(x=trend['Tháng'], y=trend['Avg Rating'], name='Rating Trung Bình', yaxis='y2', line=dict(color='#F44336', width=3)))
        
        fig.update_layout(
            title='Diễn Biến Theo Thời Gian',
            yaxis=dict(title='Review Count'),
            yaxis2=dict(title='Avg Rating', overlaying='y', side='right', range=[3.5, 5]),
            height=400,
            legend=dict(orientation='h', y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("**Insight:** Số lượng đánh giá có xu hướng tăng vào các tháng cuối năm (mùa mua sắm), trong khi Rating trung bình giữ ở mức ổn định.")


def render_section4_models():
    """Section 4: Models & Methods."""
    st.markdown('<div class="section-header">4. Mô Hình và Phương Pháp Phân Tích</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Hệ thống sử dụng kết hợp giữa **Unsupervised Learning (Clustering)** và **Large Language Models (LLM)** để hiểu sâu sắc nội dung đánh giá.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 4.1 Product Clustering (Phân Nhóm Sản Phẩm)")
        st.markdown("""
        **Vấn đề:** Dữ liệu thô có hàng ngàn sản phẩm nhưng danh mục không rõ ràng.
        
        **Giải pháp - Gemini Zero-shot Classification:**
        1. **Input:** Danh sách tên sản phẩm (Product Titles).
        2. **Process:** Sử dụng LLM (Gemini) để tự động gán nhãn danh mục dựa trên ngữ nghĩa của tên sản phẩm. Không cần training data.
        3. **Post-process:** Gom các nhóm nhỏ lẻ (<10 sản phẩm) vào nhóm "Other".
        
        **Kết quả:** Tạo ra cấu trúc danh mục rõ ràng (Electronics, Health, Home...) giúp phân tích drill-down hiệu quả.
        """)
        
    with col2:
        st.markdown("### 4.2 Aspect Analysis (Phân Tích Khía Cạnh)")
        st.markdown("""
        **Vấn đề:** Muốn biết khách hàng nói gì về cụ thể từng khía cạnh (giá, chất lượng, giao hàng).
        
        **Giải pháp - Embedding + Clustering:**
        1. **Embedding:** Sử dụng `sentence-transformers` để chuyển đổi từng review text thành vector ngữ nghĩa (384 chiều).
        2. **Dimensionality Reduction:** Dùng **UMAP** để giảm chiều dữ liệu, giữ lại cấu trúc cục bộ.
        3. **Clustering:** Dùng **KMeans** để gom nhóm các reviews có nội dung tương tự nhau -> Mỗi cụm đại diện cho một Khía Cạnh (Aspect).
        4. **Summarization:** Dùng **Gemini** để đọc các reviews trong cụm và tóm tắt nội dung chính + phân tích cảm xúc.
        """)
    
    st.markdown("### 4.3 Tại sao phương pháp này hiệu quả?")
    st.info("""
    - **Không cần dán nhãn thủ công:** Tiết kiệm thời gian và công sức.
    - **Hiểu ngữ nghĩa sâu:** Sentence Embeddings nắm bắt được ý nghĩa câu văn tốt hơn từ khóa đơn lẻ (Keyword-based). Ví dụ: "sound is tinny" sẽ được gom nhóm với "bad audio quality" dù không chung từ khóa.
    - **Tóm tắt tự nhiên:** LLM sinh ra văn bản tóm tắt dễ đọc, thay vì chỉ đưa ra đám mây từ khóa (Word Cloud).
    """)
def render_section5_results(df):
    """Section 5: Results."""
    st.markdown('<div class="section-header">5. Kết Quả Phân Tích Chi Tiết</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="subsection-header">5.1 Các Khía Cạnh Được Quan Tâm Nhất (Analyzed Aspects)</div>', unsafe_allow_html=True)
    st.write("Dưới đây là bảng tổng hợp các khía cạnh (aspects) được trích xuất từ nội dung reviews và chỉ số cảm xúc tương ứng.")
    
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
                'Khía Cạnh': aspect.capitalize(),
                'Số Lượt Đề Cập': int(mentions),
                'Tỷ Lệ (%)': mentions/len(df)*100,
                '% Tích Cực': pos_rate,
                '% Tiêu Cực': neg_rate
            })
        
        aspect_df = pd.DataFrame(aspect_data).sort_values('Số Lượt Đề Cập', ascending=False)
        
        # Display as robust dataframe with progress bars
        st.dataframe(
            aspect_df.style.format({
                'Tỷ Lệ (%)': '{:.1f}%',
                '% Tích Cực': '{:.1f}%',
                '% Tiêu Cực': '{:.1f}%',
                'Số Lượt Đề Cập': "{:,}"
            }).bar(subset=['% Tích Cực'], color='#4CAF50')
              .bar(subset=['% Tiêu Cực'], color='#F44336'),
            use_container_width=True
        )
        
        # Chart
        st.markdown("**Biểu đồ Tương Quan: Tần suất vs Cảm Xúc**")
        fig = px.scatter(
            aspect_df,
            x='Số Lượt Đề Cập',
            y='% Tiêu Cực',
            size='Số Lượt Đề Cập',
            color='Khía Cạnh',
            text='Khía Cạnh',
            title='Aspect Map: Tần suất càng lớn & Càng tiêu cực là Vấn Đề (Góc phải trên)'
        )
        fig.update_traces(textposition='top center')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Phân tích Biểu đồ:**
        - Các khía cạnh nằm ở góc **phải trên** (nhiều người nói đên & tỷ lệ tiêu cực cao) là những "pain points" cần ưu tiên giải quyết.
        - Các khía cạnh ở góc **phải dưới** (nhiều người khen & ít tiêu cực) là thế mạnh cần phát huy.
        """)


def render_section6_conclusions(df):
    """Section 6: Conclusions."""
    st.markdown('<div class="section-header">6. Kết Luận & Khuyến Nghị Chiến Lược</div>', unsafe_allow_html=True)
    
    st.markdown("Dựa trên kết quả phân tích dữ liệu và mô hình học máy, chúng tôi đề xuất các khuyến nghị sau:")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.error("**🛑 Ưu Tiên Cao (Immediate Action)**")
        st.markdown("""
        1. **Cải thiện Quy Trình Kiểm Tra Chất Lượng (Quality):**
           - `Quality` là vấn đề bị phàn nàn nhiều nhất.
           - Cần kiểm tra kỹ các lô hàng có tỷ lệ đổi trả cao.
        2. **Đào Tạo Dịch Vụ Khách Hàng:**
           - Tỷ lệ tiêu cực ở `Customer Service` đang ở mức báo động.
           - Cần cải thiện thời gian phản hồi và thái độ nhân viên.
        """)
        
    with c2:
        st.warning("**⚠️ Ưu Tiên Trung Bình (Monitor)**")
        st.markdown("""
        1. **Tối Ưu Hóa Vận Chuyển (Shipping):**
           - Khách hàng quan tâm nhiều đến tốc độ giao hàng.
           - Cần làm việc với đối tác vận chuyển để giảm thời gian ship.
        2. **Cập Nhật Mô Tả Sản Phẩm:**
           - Một số phàn nàn về việc "Not as described". Cần rà soát lại Content.
        """)
        
    with c3:
        st.success("**✅ Duy Trì & Phát Huy (Strengths)**")
        st.markdown("""
        1. **Chiến Lược Giá (Price):**
           - Khách hàng rất hài lòng về giá cả (`Price` có positive sentiment cao).
           - Có thể cân nhắc tăng nhẹ giá ở các sản phẩm premium.
        2. **Đa Dạng Hóa Danh Mục:**
           - Các ngành hàng `Electronics` và `Home` đang tăng trưởng tốt.
        """)
    
    st.markdown("---")
    st.markdown('<div class="subsection-header">6.5 Lộ Trình Phát Triển Hệ Thống (Next Steps)</div>', unsafe_allow_html=True)
    st.markdown("""
    Để nâng cao hiệu quả của hệ thống phân tích, các bước tiếp theo bao gồm:
    1. **Real-time Monitoring Dashboard:** Xây dựng dashboard theo dõi trực gian thực để phát hiện khủng hoảng truyền thông sớm.
    2. **Tích hợp thêm nguồn dữ liệu:** Kết hợp dữ liệu từ Facebook, Twitter để có cái nhìn đa chiều (Social Listening).
    3. **Fine-tune LLM:** Huấn luyện lại model Gemini trên tập dữ liệu domain-specific của Walmart để hiểu các thuật ngữ chuyên ngành tốt hơn.
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
    st.markdown(f"### Kết Quả Phân Tích: {result.get('n_aspects')} Khía Cạnh")
    
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
            
    # HIỂN THỊ TÓM TẮT CHUNG (MỚI)
    if result.get('overall_summary'):
        st.info(f"**Tóm Tắt Tổng Quan:**\n\n{result.get('overall_summary')}")
    
    st.markdown("---")
    st.markdown("### Chi Tiết Từng Khía Cạnh")
    
    # Hiển thị từng khía cạnh
    for aspect in result.get('aspects', []):
        with st.container():
            st.markdown(f"#### Khía cạnh {aspect['aspect_id']}: {aspect['aspect_name']}")
            
            # Layout: Metrics bên trái, Summary bên phải
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Số Reviews Discussed", aspect['review_count'])
                
                # Sentiment Chart nhỏ
                sentiment = aspect.get('sentiment', {})
                sent_data = {
                    'Positive': sentiment.get('positive_pct', 0),
                    'Neutral': sentiment.get('neutral_pct', 0),
                    'Negative': sentiment.get('negative_pct', 0)
                }
                st.write("**Cảm xúc:**")
                st.progress(sent_data['Positive']/100, text=f"Positive: {sent_data['Positive']}%")
                st.progress(sent_data['Negative']/100, text=f"Negative: {sent_data['Negative']}%")
            
            with col2:
                st.markdown(f"**Tóm tắt đại diện:**")
                st.success(aspect['summary'])
            
            # Sample reviews
            if aspect.get('sample_reviews'):
                with st.expander(f"Xem đánh giá chi tiết về {aspect['aspect_name']}"):
                    for i, review in enumerate(aspect['sample_reviews'][:5], 1):
                        st.markdown(f"**{i}.** {review}")
            
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
    st.markdown(f"### Kết Quả Phân Tích: Khía Cạnh \"{result.get('aspect_name')}\"")
    
    # Overview Summary (Cái chung nhất)
    # st.markdown("#### 1. Tổng Quan")
    st.info(f"**Tóm Tắt Khía Cạnh:**\n\n{result.get('summary', 'Không có tóm tắt')}")
    
    # Sentiment Analysis Overview
    sentiment = result.get('sentiment', {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sentiment Positive", f"{sentiment.get('positive_pct', 0)}%")
    with col2:
        st.metric("Sentiment Neutral", f"{sentiment.get('neutral_pct', 0)}%")
    with col3:
        st.metric("Sentiment Negative", f"{sentiment.get('negative_pct', 0)}%")
        
    st.markdown("---")
    
    # Details (Cái riêng)
    st.markdown("#### 2. Chi Tiết Phân Tích")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Tổng Reviews Đã Quét", f"{result.get('total_reviews_analyzed', 0):,}")
    with col2:
        st.metric("Reviews Liên Quan Found", f"{result.get('relevant_reviews_count', 0):,}")
        
    st.markdown("#### Các Đánh Giá Điển Hình Nhất")
    st.caption("Các đánh giá được sắp xếp theo độ tương đồng ngữ nghĩa (Semantic Similarity)")
    
    sample_reviews = result.get('sample_reviews', [])
    if sample_reviews:
        for i, item in enumerate(sample_reviews, 1):
            review = item.get('review', '')
            similarity = item.get('similarity', 0)
            
            with st.container():
                st.markdown(f"**Review #{i}** (Similarity: {similarity:.3f})")
                st.markdown(f"> {review}")
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


# ============================================================
# RAG QUERY PAGE
# ============================================================

def render_rag_query(df):
    """Render RAG query page."""
    st.markdown('<div class="main-header">RAG Query - Truy Vấn Nhanh</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Sử dụng RAG (Retrieval-Augmented Generation) để truy vấn nhanh các khía cạnh đã được pre-compute.
    
    **Workflow:**
    1. Pre-compute: Phân tích tất cả categories và lưu vào vector store
    2. Query: Tìm kiếm semantic và generate response với LLM
    """)
    
    st.markdown("---")
    
    # Initialize components
    try:
        from src.analysis.rag_pipeline import create_rag_pipeline
        vector_store, query_engine, precompute_pipeline = create_rag_pipeline(df)
    except Exception as e:
        st.error(f"Không thể khởi tạo RAG pipeline: {e}")
        return
    
    # Stats
    stats = vector_store.get_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Documents", stats.get('n_documents', 0))
    with col2:
        st.metric("Categories", stats.get('n_categories', 0))
    with col3:
        st.metric("Status", stats.get('status', 'N/A'))
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2 = st.tabs(["Query", "Pre-Compute"])
    
    with tab1:
        st.markdown("### Truy Vấn")
        
        if stats.get('n_documents', 0) == 0:
            st.warning("Chưa có dữ liệu. Hãy chạy Pre-Compute trước.")
        else:
            query = st.text_input(
                "Nhập câu hỏi:", 
                placeholder="Ví dụ: Khách hàng nghĩ gì về chất lượng âm thanh của tai nghe?"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                top_k = st.slider("Số kết quả:", 1, 10, 5)
            with col2:
                use_llm = st.checkbox("Sử dụng LLM", value=True)
            
            if st.button("Tìm Kiếm", type="primary"):
                if query:
                    with st.spinner("Đang tìm kiếm..."):
                        result = query_engine.query(query, top_k=top_k, use_llm=use_llm)
                        
                        if result.get('success'):
                            st.markdown("### Trả Lời")
                            st.markdown(result['response'])
                            
                            st.markdown("---")
                            st.markdown("### Aspects Liên Quan")
                            for asp in result.get('retrieved_aspects', []):
                                with st.expander(f"{asp['aspect_name']} ({asp['category']}) - Similarity: {asp['similarity']:.3f}"):
                                    st.markdown(f"**Summary:** {asp['summary']}")
                                    sentiment = asp.get('sentiment', {})
                                    st.markdown(f"**Sentiment:** {sentiment.get('interpretation', 'N/A')}")
                        else:
                            st.error(result.get('error', 'Query thất bại'))
                else:
                    st.warning("Vui lòng nhập câu hỏi")
    
    with tab2:
        st.markdown("### Pre-Compute Aspects")
        st.markdown("Phân tích tất cả categories và lưu vào vector store.")
        
        col1, col2 = st.columns(2)
        with col1:
            n_aspects = st.number_input("Số aspects per category:", 3, 10, 5, key="rag_n_aspects")
        with col2:
            max_cats = st.number_input("Max categories (0=all):", 0, 100, 0, key="rag_max_cats")
        
        if st.button("Chạy Pre-Compute", type="primary", key="run_precompute"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total, category):
                progress_bar.progress(current / total)
                status_text.text(f"[{current}/{total}] {category}")
            
            with st.spinner("Đang pre-compute..."):
                result = precompute_pipeline.run_full_pipeline(
                    n_aspects=int(n_aspects),
                    max_categories=int(max_cats) if max_cats > 0 else None,
                    progress_callback=update_progress
                )
                
                if result.get('success'):
                    st.success(f"Thành công! Đã xử lý {result['n_processed']} categories, {result['total_documents']} documents.")
                    
                    if result.get('failed_categories'):
                        st.warning(f"Thất bại: {len(result['failed_categories'])} categories")
                else:
                    st.error(result.get('error', 'Pre-compute thất bại'))


def render_full_evaluation(df):
    """Render full dataset evaluation section."""
    st.markdown("---")
    st.markdown("### Đánh Giá Toàn Bộ Dataset")
    
    col1, col2 = st.columns(2)
    with col1:
        n_aspects = st.number_input("Số aspects:", 3, 10, 5, key="full_eval_aspects")
    with col2:
        max_cats = st.number_input("Max categories (0=all):", 0, 50, 10, key="full_eval_cats")
    
    if st.button("Chạy Full Evaluation", type="secondary", key="run_full_eval"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total, category):
            progress_bar.progress(current / total)
            status_text.text(f"[{current}/{total}] {category}")
        
        with st.spinner("Đang đánh giá toàn bộ dataset... (có thể mất vài phút)"):
            try:
                from src.analysis.evaluator import full_dataset_evaluation
                
                result = full_dataset_evaluation(
                    df,
                    n_aspects=int(n_aspects),
                    max_categories=int(max_cats) if max_cats > 0 else None,
                    progress_callback=update_progress
                )
                
                if result.get('success'):
                    display_full_evaluation_results(result)
                else:
                    st.error(result.get('error', 'Đánh giá thất bại'))
                    
            except Exception as e:
                st.error(f"Lỗi: {e}")


def display_full_evaluation_results(result: dict):
    """Display full evaluation results."""
    st.markdown("---")
    st.markdown("## Kết Quả Đánh Giá Toàn Dataset")
    
    overall = result.get('overall', {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Score", f"{overall.get('score', 0):.1%}")
    with col2:
        st.metric("Grade", overall.get('grade', 'N/A'))
    with col3:
        st.metric("Categories", result.get('categories_evaluated', 0))
    with col4:
        st.metric("Failed", result.get('categories_failed', 0))
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Silhouette", f"{overall.get('avg_silhouette', 0):.4f}")
    with col2:
        st.metric("Avg Coherence", f"{overall.get('avg_coherence', 0):.4f}")
    with col3:
        st.metric("Avg Coverage", f"{overall.get('avg_coverage', 0):.1f}%")
    
    # Best/Worst categories
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Top 5 Categories")
        for cat in result.get('best_categories', []):
            st.markdown(f"- **{cat['category']}**: {cat['overall_score']:.1%} ({cat['grade']})")
    
    with col2:
        st.markdown("### Bottom 5 Categories")
        for cat in result.get('worst_categories', []):
            st.markdown(f"- **{cat['category']}**: {cat['overall_score']:.1%} ({cat['grade']})")
    
    # Details table
    with st.expander("Chi tiết theo Category"):
        details_df = pd.DataFrame(result.get('category_details', []))
        if not details_df.empty:
            st.dataframe(details_df[['category', 'n_reviews', 'silhouette', 'coherence', 'coverage', 'overall_score', 'grade']], 
                        use_container_width=True, hide_index=True)


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
    elif page == "RAG Query":
        render_rag_query(df)
    elif page == "Danh Gia Mo Hinh":
        render_evaluation(df, summarizer)
        render_full_evaluation(df)


if __name__ == "__main__":
    main()

