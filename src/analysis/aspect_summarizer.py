"""
Aspect Summarizer sử dụng Embedding và Clustering.

Phương pháp:
- Case 1: Nhập sản phẩm + số khía cạnh -> embedding -> UMAP -> clustering -> đặt tên -> summarize
- Case 2: Nhập sản phẩm + tên khía cạnh -> embedding similarity -> trace reviews -> summarize

Không sử dụng rule-based, hoàn toàn dựa trên semantic embeddings.

OPTIMIZATIONS:
- Caching embeddings to disk
- Batch processing với progress bar
- FastMode với smaller model
"""
from __future__ import annotations

import logging
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path(__file__).parent.parent.parent / "outputs" / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class EmbeddingAspectSummarizer:
    """
    Tóm tắt theo khía cạnh sử dụng Embeddings và Clustering.
    
    OPTIMIZATIONS:
    - Caching embeddings to disk (không cần tính lại)
    - Batch encoding với parallel processing
    - FastMode sử dụng model nhỏ hơn
    
    Workflow Case 1 (Số khía cạnh):
    1. Lọc reviews theo sản phẩm/category
    2. Tạo embeddings cho tất cả reviews (CACHED)
    3. Giảm chiều bằng UMAP/PCA
    4. Gom cụm bằng KMeans với k = số khía cạnh mong muốn
    5. Đặt tên cho từng cluster bằng LLM
    6. Summarize từng khía cạnh bằng LLM
    
    Workflow Case 2 (Tên khía cạnh):
    1. Lọc reviews theo sản phẩm/category  
    2. Tạo embeddings (CACHED)
    3. Tính similarity
    4. Summarize
    """
    
    # Model options: faster to slower
    MODEL_OPTIONS = {
        'fast': 'all-MiniLM-L6-v2',      # 384 dim, fastest
        'balanced': 'paraphrase-MiniLM-L3-v2',  # 384 dim, fast
        'accurate': 'all-mpnet-base-v2',  # 768 dim, accurate but slow
    }
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        model_name: str = 'all-MiniLM-L6-v2',
        fast_mode: bool = True,
        use_cache: bool = True
    ):
        """
        Khởi tạo EmbeddingAspectSummarizer.
        
        Args:
            df: DataFrame chứa reviews
            model_name: Tên model sentence-transformers
            fast_mode: Sử dụng model nhanh hơn
            use_cache: Cache embeddings vào disk
        """
        self.df = df.copy()
        self.use_cache = use_cache
        
        # Chọn model dựa trên fast_mode
        if fast_mode:
            self.model_name = self.MODEL_OPTIONS['fast']
        else:
            self.model_name = model_name
            
        self.embedding_model = None
        self.gemini_client = None
        self._embeddings_cache = {}  # In-memory cache
        
        self._init_models()
        
    def _init_models(self) -> None:
        """Khởi tạo các models cần thiết."""
        # Sentence Transformers
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Đang tải embedding model: {self.model_name}...")
            self.embedding_model = SentenceTransformer(self.model_name)
            logger.info(f"Đã tải xong embedding model!")
        except ImportError:
            logger.error("Cần cài đặt: pip install sentence-transformers")
        except Exception as e:
            logger.error(f"Lỗi tải embedding model: {e}")
            
        # Gemini Client
        try:
            from src.clustering.gemini_client import GeminiClient
            self.gemini_client = GeminiClient()
            if not self.gemini_client.is_available:
                logger.warning("Gemini API không khả dụng")
                self.gemini_client = None
        except Exception as e:
            logger.warning(f"Không thể khởi tạo Gemini: {e}")
            self.gemini_client = None
    
    def _get_cache_key(self, texts: List[str]) -> str:
        """Tạo cache key từ danh sách texts."""
        content = ''.join(sorted(texts[:100]))  # Chỉ dùng 100 đầu tiên cho key
        return hashlib.md5(f"{content}_{self.model_name}".encode()).hexdigest()
    
    def _load_embeddings_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embeddings từ disk cache."""
        if not self.use_cache:
            return None
            
        cache_path = CACHE_DIR / f"embeddings_{cache_key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    logger.info(f"Đang load embeddings từ cache...")
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Không thể load cache: {e}")
        return None
    
    def _save_embeddings_to_cache(self, cache_key: str, embeddings: np.ndarray) -> None:
        """Save embeddings vào disk cache."""
        if not self.use_cache:
            return
            
        cache_path = CACHE_DIR / f"embeddings_{cache_key}.pkl"
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info(f"Đã cache embeddings ({embeddings.shape})")
        except Exception as e:
            logger.warning(f"Không thể save cache: {e}")
    
    def _get_reviews_for_product(
        self, 
        product: Optional[str] = None,
        category: Optional[str] = None,
        max_reviews: int = 500
    ) -> pd.DataFrame:
        """
        Lọc reviews theo sản phẩm hoặc danh mục.
        
        Args:
            product: Tên sản phẩm (tìm trong title)
            category: Tên danh mục
            max_reviews: Số lượng reviews tối đa
            
        Returns:
            DataFrame đã lọc
        """
        df = self.df.copy()
        
        # Lọc reviews có nội dung
        if 'review' in df.columns:
            df = df[df['review'].notna() & (df['review'] != '')]
        else:
            return pd.DataFrame()
        
        # Lọc theo category
        if category and 'product_category' in df.columns:
            df = df[df['product_category'].str.contains(category, case=False, na=False)]
            
        # Lọc theo product name
        if product and 'title' in df.columns:
            df = df[df['title'].str.contains(product, case=False, na=False)]
            
        # Giới hạn số lượng
        if len(df) > max_reviews:
            df = df.sample(n=max_reviews, random_state=42)
            
        return df
    
    def _create_embeddings(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Tạo embeddings cho danh sách văn bản với caching.
        
        Args:
            texts: Danh sách văn bản
            batch_size: Số texts xử lý mỗi batch (tăng nếu có GPU)
            
        Returns:
            Ma trận embeddings (n_texts, embedding_dim)
        """
        if self.embedding_model is None:
            raise ValueError("Embedding model chưa được khởi tạo. Cài đặt: pip install sentence-transformers")
        
        # Check cache
        cache_key = self._get_cache_key(texts)
        cached = self._load_embeddings_from_cache(cache_key)
        if cached is not None and len(cached) == len(texts):
            return cached
        
        # Create embeddings với batch processing
        logger.info(f"Đang tạo embeddings cho {len(texts)} texts (batch_size={batch_size})...")
        embeddings = self.embedding_model.encode(
            texts, 
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=batch_size
        )
        
        # Save to cache
        self._save_embeddings_to_cache(cache_key, embeddings)
        
        return embeddings
    
    def _reduce_dimensions(self, embeddings: np.ndarray, n_components: int = 5) -> np.ndarray:
        """
        Giảm chiều embeddings bằng UMAP.
        
        Args:
            embeddings: Ma trận embeddings
            n_components: Số chiều đầu ra
            
        Returns:
            Ma trận đã giảm chiều
        """
        try:
            from umap import UMAP
            reducer = UMAP(
                n_components=n_components,
                n_neighbors=min(15, len(embeddings) - 1),
                min_dist=0.1,
                metric='cosine',
                random_state=42
            )
            reduced = reducer.fit_transform(embeddings)
            return reduced
        except ImportError:
            logger.warning("UMAP không khả dụng, sử dụng PCA")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(n_components, embeddings.shape[1]))
            return pca.fit_transform(embeddings)
    
    def _cluster_embeddings(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Gom cụm embeddings bằng KMeans.
        
        Args:
            embeddings: Ma trận embeddings (đã giảm chiều)
            n_clusters: Số cụm mong muốn
            
        Returns:
            Array labels cho từng document
        """
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        labels = kmeans.fit_predict(embeddings)
        return labels
    
    def _name_cluster(self, reviews: List[str]) -> str:
        """
        Đặt tên cho cluster bằng LLM.
        
        Args:
            reviews: Danh sách reviews trong cluster
            
        Returns:
            Tên khía cạnh
        """
        if self.gemini_client is None:
            # Fallback: lấy từ phổ biến nhất
            from collections import Counter
            words = ' '.join(reviews[:10]).lower().split()
            common = Counter(words).most_common(3)
            return ' / '.join([w for w, _ in common if len(w) > 3])
        
        sample_reviews = reviews[:10]
        reviews_text = '\n'.join([f"- {r[:150]}" for r in sample_reviews])
        
        prompt = f"""Phân tích các đánh giá sau và đặt TÊN NGẮN GỌN (2-3 từ) cho khía cạnh chung mà chúng đề cập:

Các đánh giá:
{reviews_text}

Yêu cầu:
- Tên phải ngắn gọn (2-3 từ)
- Phản ánh khía cạnh chung của các đánh giá
- Ví dụ: "Chất lượng âm thanh", "Thời gian giao hàng", "Giá trị sản phẩm"

Tên khía cạnh:"""

        try:
            response = self.gemini_client.generate(prompt, max_tokens=50)
            return response.strip().strip('"').strip("'") if response else "Khía cạnh chung"
        except Exception:
            return "Khía cạnh chung"
    
    def _summarize_reviews(self, aspect_name: str, reviews: List[str]) -> str:
        """
        Tóm tắt các reviews của một khía cạnh bằng LLM.
        
        Args:
            aspect_name: Tên khía cạnh
            reviews: Danh sách reviews
            
        Returns:
            Bản tóm tắt
        """
        if self.gemini_client is None:
            return f"Có {len(reviews)} đánh giá về {aspect_name}."
        
        sample_reviews = reviews[:20]
        reviews_text = '\n'.join([f"- {r[:200]}" for r in sample_reviews])
        
        prompt = f"""Tóm tắt những gì khách hàng nói về "{aspect_name}" dựa trên các đánh giá sau:

Các đánh giá:
{reviews_text}

Yêu cầu:
- Viết bằng tiếng Việt
- Tóm tắt ngắn gọn 2-3 câu
- Nêu rõ ý kiến chung (tích cực/tiêu cực)
- Đề cập các điểm cụ thể được nhắc đến

Tóm tắt:"""

        try:
            response = self.gemini_client.generate(prompt, max_tokens=300)
            return response.strip() if response else f"Có {len(reviews)} đánh giá về {aspect_name}."
        except Exception as e:
            logger.error(f"Lỗi summarize: {e}")
            return f"Có {len(reviews)} đánh giá về {aspect_name}."
    
    def analyze_by_num_aspects(
        self,
        n_aspects: int,
        product: Optional[str] = None,
        category: Optional[str] = None,
        max_reviews: int = 500
    ) -> Dict[str, Any]:
        """
        Case 1: Phân tích theo số khía cạnh mong muốn.
        
        Workflow:
        1. Lọc reviews theo sản phẩm/category
        2. Tạo embeddings
        3. Giảm chiều bằng UMAP
        4. Gom cụm bằng KMeans với k = n_aspects
        5. Đặt tên cho từng cluster
        6. Summarize từng khía cạnh
        
        Args:
            n_aspects: Số khía cạnh mong muốn
            product: Tên sản phẩm (tùy chọn)
            category: Tên danh mục (tùy chọn)
            max_reviews: Số reviews tối đa
            
        Returns:
            Kết quả phân tích với các khía cạnh và tóm tắt
        """
        logger.info(f"Phân tích {n_aspects} khía cạnh cho {product or category or 'tất cả sản phẩm'}")
        
        # Bước 1: Lọc reviews
        filtered_df = self._get_reviews_for_product(product, category, max_reviews)
        
        if len(filtered_df) < n_aspects:
            return {
                'success': False,
                'error': f"Không đủ reviews ({len(filtered_df)}). Cần ít nhất {n_aspects} reviews.",
                'product': product,
                'category': category
            }
        
        reviews = filtered_df['review'].tolist()
        logger.info(f"Đã lọc được {len(reviews)} reviews")
        
        # Bước 2: Tạo embeddings
        logger.info("Đang tạo embeddings...")
        embeddings = self._create_embeddings(reviews)
        
        # Bước 3: Giảm chiều
        logger.info("Đang giảm chiều...")
        n_components = min(10, len(reviews) - 1, embeddings.shape[1])
        reduced = self._reduce_dimensions(embeddings, n_components=n_components)
        
        # Bước 4: Gom cụm
        logger.info(f"Đang gom cụm thành {n_aspects} nhóm...")
        labels = self._cluster_embeddings(reduced, n_clusters=n_aspects)
        
        # Bước 5, 6, 7: Đặt tên, phân tích sentiment, và summarize từng cluster
        aspects_result = []
        
        for cluster_id in range(n_aspects):
            cluster_mask = labels == cluster_id
            cluster_reviews = [reviews[i] for i in range(len(reviews)) if cluster_mask[i]]
            
            if not cluster_reviews:
                continue
                
            # Đặt tên
            logger.info(f"Đang đặt tên cho cluster {cluster_id + 1}...")
            aspect_name = self._name_cluster(cluster_reviews)
            
            # Phân tích sentiment
            logger.info(f"Đang phân tích sentiment cho cluster {cluster_id + 1}...")
            sentiment = self._analyze_sentiment(cluster_reviews)
            
            # Summarize
            logger.info(f"Đang tóm tắt cluster {cluster_id + 1}...")
            summary = self._summarize_reviews(aspect_name, cluster_reviews)
            
            aspects_result.append({
                'aspect_id': cluster_id + 1,
                'aspect_name': aspect_name,
                'review_count': len(cluster_reviews),
                'sentiment': sentiment,
                'summary': summary,
                'sample_reviews': cluster_reviews[:5]
            })
        
        return {
            'success': True,
            'product': product,
            'category': category,
            'total_reviews': len(reviews),
            'n_aspects': n_aspects,
            'aspects': aspects_result
        }
    
    def _analyze_sentiment(self, reviews: List[str]) -> Dict[str, Any]:
        """
        Phân tích sentiment cho danh sách reviews.
        
        Args:
            reviews: Danh sách reviews
            
        Returns:
            Dictionary với sentiment scores và interpretation
        """
        # Keyword-based sentiment analysis
        positive_keywords = [
            'great', 'excellent', 'amazing', 'love', 'perfect', 'good', 'awesome', 
            'best', 'fantastic', 'wonderful', 'happy', 'satisfied', 'recommend',
            'tốt', 'hay', 'đẹp', 'thích', 'xuất sắc', 'tuyệt vời'
        ]
        negative_keywords = [
            'bad', 'terrible', 'horrible', 'hate', 'worst', 'awful', 'poor',
            'broken', 'disappointed', 'waste', 'useless', 'defective', 'garbage',
            'tệ', 'xấu', 'dở', 'hỏng', 'thất vọng', 'không tốt'
        ]
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for review in reviews:
            review_lower = review.lower()
            
            has_positive = any(kw in review_lower for kw in positive_keywords)
            has_negative = any(kw in review_lower for kw in negative_keywords)
            
            if has_positive and not has_negative:
                positive_count += 1
            elif has_negative and not has_positive:
                negative_count += 1
            elif has_positive and has_negative:
                # Mixed - count as neutral
                neutral_count += 1
            else:
                neutral_count += 1
        
        total = len(reviews)
        
        positive_pct = round(positive_count / total * 100, 1) if total > 0 else 0
        negative_pct = round(negative_count / total * 100, 1) if total > 0 else 0
        neutral_pct = round(neutral_count / total * 100, 1) if total > 0 else 0
        
        # Determine overall sentiment
        if positive_pct >= 60:
            overall = 'positive'
            interpretation = 'Đa số đánh giá tích cực'
        elif negative_pct >= 40:
            overall = 'negative'
            interpretation = 'Nhiều đánh giá tiêu cực'
        elif positive_pct > negative_pct:
            overall = 'mixed_positive'
            interpretation = 'Hơi nghiêng về tích cực'
        elif negative_pct > positive_pct:
            overall = 'mixed_negative'
            interpretation = 'Hơi nghiêng về tiêu cực'
        else:
            overall = 'neutral'
            interpretation = 'Đánh giá trung lập'
        
        return {
            'overall': overall,
            'interpretation': interpretation,
            'positive_pct': positive_pct,
            'negative_pct': negative_pct,
            'neutral_pct': neutral_pct,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count
        }
    
    def analyze_by_aspect_name(
        self,
        aspect_name: str,
        product: Optional[str] = None,
        category: Optional[str] = None,
        max_reviews: int = 500,
        similarity_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Case 2: Phân tích theo tên khía cạnh.
        
        Workflow:
        1. Lọc reviews theo sản phẩm/category
        2. Tạo embeddings cho reviews
        3. Tạo embedding cho tên khía cạnh
        4. Tính similarity giữa reviews và khía cạnh
        5. Lọc reviews có similarity cao
        6. Summarize các reviews đó
        
        Args:
            aspect_name: Tên khía cạnh muốn phân tích
            product: Tên sản phẩm (tùy chọn)
            category: Tên danh mục (tùy chọn)
            max_reviews: Số reviews tối đa
            similarity_threshold: Ngưỡng similarity tối thiểu
            
        Returns:
            Kết quả phân tích với tóm tắt
        """
        logger.info(f"Phân tích khía cạnh '{aspect_name}' cho {product or category or 'tất cả sản phẩm'}")
        
        # Bước 1: Lọc reviews
        filtered_df = self._get_reviews_for_product(product, category, max_reviews)
        
        if len(filtered_df) == 0:
            return {
                'success': False,
                'error': "Không tìm thấy reviews phù hợp.",
                'aspect_name': aspect_name,
                'product': product,
                'category': category
            }
        
        reviews = filtered_df['review'].tolist()
        logger.info(f"Đã lọc được {len(reviews)} reviews")
        
        # Bước 2: Tạo embeddings cho reviews
        logger.info("Đang tạo embeddings cho reviews...")
        review_embeddings = self._create_embeddings(reviews)
        
        # Bước 3: Tạo embedding cho tên khía cạnh
        aspect_embedding = self._create_embeddings([aspect_name])[0]
        
        # Bước 4: Tính cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([aspect_embedding], review_embeddings)[0]
        
        # Bước 5: Lọc reviews có similarity cao
        relevant_indices = np.where(similarities >= similarity_threshold)[0]
        relevant_reviews = [reviews[i] for i in relevant_indices]
        relevant_similarities = [similarities[i] for i in relevant_indices]
        
        if not relevant_reviews:
            # Nếu không có review nào đạt ngưỡng, lấy top 20 reviews gần nhất
            top_indices = np.argsort(similarities)[-20:][::-1]
            relevant_reviews = [reviews[i] for i in top_indices]
            relevant_similarities = [similarities[i] for i in top_indices]
        
        logger.info(f"Tìm thấy {len(relevant_reviews)} reviews liên quan đến '{aspect_name}'")
        
        # Bước 6: Summarize
        summary = self._summarize_reviews(aspect_name, relevant_reviews)
        
        # Tính sentiment đơn giản
        positive_keywords = ['tốt', 'hay', 'đẹp', 'great', 'good', 'excellent', 'love', 'amazing']
        negative_keywords = ['tệ', 'xấu', 'dở', 'bad', 'terrible', 'hate', 'poor', 'broken']
        
        positive_count = sum(1 for r in relevant_reviews 
                          if any(kw in r.lower() for kw in positive_keywords))
        negative_count = sum(1 for r in relevant_reviews 
                          if any(kw in r.lower() for kw in negative_keywords))
        
        total = len(relevant_reviews)
        
        return {
            'success': True,
            'aspect_name': aspect_name,
            'product': product,
            'category': category,
            'total_reviews_analyzed': len(reviews),
            'relevant_reviews_count': len(relevant_reviews),
            'avg_similarity': float(np.mean(relevant_similarities)),
            'sentiment': {
                'positive_pct': round(positive_count / total * 100, 1) if total > 0 else 0,
                'negative_pct': round(negative_count / total * 100, 1) if total > 0 else 0,
                'neutral_pct': round((total - positive_count - negative_count) / total * 100, 1) if total > 0 else 0
            },
            'summary': summary,
            'sample_reviews': [
                {'review': relevant_reviews[i], 'similarity': round(relevant_similarities[i], 3)}
                for i in range(min(5, len(relevant_reviews)))
            ]
        }


# Giữ lại class cũ cho backward compatibility
class AspectSummarizer:
    """Wrapper class để tương thích với code cũ."""
    
    def __init__(self, df: pd.DataFrame, topic_modeler=None):
        self.df = df
        self.embedding_summarizer = None
        self._init_embedding_summarizer()
        
    def _init_embedding_summarizer(self):
        try:
            self.embedding_summarizer = EmbeddingAspectSummarizer(self.df)
        except Exception as e:
            logger.warning(f"Không thể khởi tạo EmbeddingAspectSummarizer: {e}")
    
    def summarize_aspect(
        self,
        aspect: str,
        category: Optional[str] = None,
        max_reviews: int = 100
    ) -> Dict[str, Any]:
        """Tóm tắt khía cạnh (tương thích code cũ)."""
        if self.embedding_summarizer:
            result = self.embedding_summarizer.analyze_by_aspect_name(
                aspect_name=aspect,
                category=category,
                max_reviews=max_reviews
            )
            
            if result['success']:
                return {
                    'aspect': aspect,
                    'category': category,
                    'review_count': result['relevant_reviews_count'],
                    'summary': result['summary'],
                    'sentiment': 'positive' if result['sentiment']['positive_pct'] > 50 else 
                                'negative' if result['sentiment']['negative_pct'] > 30 else 'neutral',
                    'sentiment_scores': result['sentiment'],
                    'key_points': [],
                    'sample_reviews': [r['review'] for r in result.get('sample_reviews', [])]
                }
        
        # Fallback
        return {
            'aspect': aspect,
            'category': category,
            'review_count': 0,
            'summary': f"Không thể phân tích khía cạnh '{aspect}'.",
            'sentiment': 'neutral',
            'sentiment_scores': {'positive': 0, 'negative': 0, 'neutral': 100},
            'key_points': [],
            'sample_reviews': []
        }
    
    def get_top_aspects_for_category(
        self,
        category: str,
        n_aspects: int = 5
    ) -> Dict[str, Any]:
        """Lấy top N khía cạnh cho danh mục (tương thích code cũ)."""
        if self.embedding_summarizer:
            result = self.embedding_summarizer.analyze_by_num_aspects(
                n_aspects=n_aspects,
                category=category
            )
            
            if result['success']:
                aspects = []
                for asp in result['aspects']:
                    aspects.append({
                        'aspect': asp['aspect_name'],
                        'review_count': asp['review_count'],
                        'summary': asp['summary'],
                        'sentiment': 'neutral',
                        'sentiment_scores': {'positive': 0, 'negative': 0, 'neutral': 100},
                        'key_points': [],
                        'sample_reviews': asp.get('sample_reviews', [])
                    })
                
                return {
                    'category': category,
                    'total_reviews': result['total_reviews'],
                    'aspects': aspects
                }
        
        # Fallback
        return {
            'category': category,
            'total_reviews': 0,
            'error': "Không thể phân tích.",
            'aspects': []
        }
