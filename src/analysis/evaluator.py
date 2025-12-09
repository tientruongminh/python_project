"""
Evaluation Metrics cho Aspect-Based Summarization System.

Các loại metrics:
1. Clustering Quality: Silhouette, Calinski-Harabasz, Davies-Bouldin
2. Topic Coherence: Semantic similarity trong từng cluster
3. Coverage: Tỷ lệ reviews được gán aspect
4. Summary Quality: Semantic similarity giữa summary và reviews
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from collections import Counter

logger = logging.getLogger(__name__)


class AspectModelEvaluator:
    """
    Đánh giá chất lượng mô hình Aspect-Based Summarization.
    
    Metrics được chia thành 4 nhóm:
    1. Clustering Quality - Đánh giá chất lượng gom cụm
    2. Topic Coherence - Đánh giá độ mạch lạc của topics
    3. Coverage - Đánh giá độ bao phủ
    4. Summary Quality - Đánh giá chất lượng tóm tắt
    """
    
    def __init__(self, embedding_model=None):
        """
        Khởi tạo evaluator.
        
        Args:
            embedding_model: Model sentence-transformers (optional)
        """
        self.embedding_model = embedding_model
        
    def _ensure_embedding_model(self):
        """Load embedding model nếu chưa có."""
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                logger.error("Cần cài đặt: pip install sentence-transformers")
                return False
        return True
    
    # ========================================
    # 1. CLUSTERING QUALITY METRICS
    # ========================================
    
    def evaluate_clustering(
        self, 
        embeddings: np.ndarray, 
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Đánh giá chất lượng clustering.
        
        Metrics:
        - Silhouette Score [-1, 1]: Cao hơn = tốt hơn, clusters rõ ràng
        - Calinski-Harabasz Index: Cao hơn = tốt hơn, clusters dense và well-separated
        - Davies-Bouldin Index: Thấp hơn = tốt hơn, clusters compact
        
        Args:
            embeddings: Ma trận embeddings (n_samples, n_features)
            labels: Cluster labels cho từng sample
            
        Returns:
            Dictionary với các metrics
        """
        from sklearn.metrics import (
            silhouette_score, 
            calinski_harabasz_score, 
            davies_bouldin_score
        )
        
        # Cần ít nhất 2 clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 2:
            return {
                'silhouette_score': 0.0,
                'calinski_harabasz': 0.0,
                'davies_bouldin': float('inf'),
                'n_clusters': n_clusters,
                'interpretation': 'Không đủ clusters để đánh giá'
            }
        
        # Tính metrics
        silhouette = silhouette_score(embeddings, labels)
        calinski = calinski_harabasz_score(embeddings, labels)
        davies = davies_bouldin_score(embeddings, labels)
        
        # Interpretation
        if silhouette >= 0.5:
            silhouette_interp = "Tốt - clusters rõ ràng"
        elif silhouette >= 0.25:
            silhouette_interp = "Trung bình - có overlap"
        else:
            silhouette_interp = "Yếu - clusters không rõ ràng"
            
        return {
            'silhouette_score': round(silhouette, 4),
            'silhouette_interpretation': silhouette_interp,
            'calinski_harabasz': round(calinski, 2),
            'davies_bouldin': round(davies, 4),
            'davies_interpretation': 'Tốt' if davies < 1.0 else 'Trung bình' if davies < 2.0 else 'Yếu',
            'n_clusters': n_clusters
        }
    
    # ========================================
    # 2. TOPIC COHERENCE METRICS
    # ========================================
    
    def evaluate_topic_coherence(
        self, 
        clusters: Dict[int, List[str]],
        top_n_words: int = 10
    ) -> Dict[str, Any]:
        """
        Đánh giá độ mạch lạc của topics/aspects.
        
        Coherence cao = các reviews trong cùng cluster nói về cùng chủ đề.
        
        Methods:
        - Intra-cluster similarity: Similarity trung bình trong cluster
        - Top words overlap: Độ overlap giữa top words
        
        Args:
            clusters: Dict mapping cluster_id -> list of reviews
            top_n_words: Số từ phổ biến nhất để so sánh
            
        Returns:
            Dictionary với coherence metrics
        """
        if not self._ensure_embedding_model():
            return {'error': 'Không có embedding model'}
        
        coherence_scores = []
        cluster_details = []
        
        for cluster_id, reviews in clusters.items():
            if len(reviews) < 2:
                continue
                
            # Sample để tính nhanh
            sample_reviews = reviews[:50] if len(reviews) > 50 else reviews
            
            # Tính embeddings
            embeddings = self.embedding_model.encode(sample_reviews, show_progress_bar=False)
            
            # Tính intra-cluster similarity
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(embeddings)
            
            # Lấy upper triangle (không tính diagonal)
            upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
            avg_similarity = float(np.mean(upper_tri))
            
            # Top words
            all_words = ' '.join(sample_reviews).lower().split()
            word_counts = Counter(all_words)
            # Loại bỏ stopwords đơn giản
            stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'it', 'i', 'and', 'to', 'for', 'of', 'in', 'on', 'this', 'that', 'with'}
            top_words = [w for w, c in word_counts.most_common(top_n_words + 20) if w not in stopwords and len(w) > 2][:top_n_words]
            
            coherence_scores.append(avg_similarity)
            cluster_details.append({
                'cluster_id': cluster_id,
                'n_reviews': len(reviews),
                'avg_similarity': round(avg_similarity, 4),
                'top_words': top_words[:5]
            })
        
        overall_coherence = float(np.mean(coherence_scores)) if coherence_scores else 0.0
        
        # Interpretation
        if overall_coherence >= 0.6:
            interp = "Rất tốt - Topics mạch lạc, rõ ràng"
        elif overall_coherence >= 0.4:
            interp = "Tốt - Topics có ý nghĩa"
        elif overall_coherence >= 0.25:
            interp = "Trung bình - Topics có overlap"
        else:
            interp = "Yếu - Topics không rõ ràng"
        
        return {
            'overall_coherence': round(overall_coherence, 4),
            'interpretation': interp,
            'n_clusters_evaluated': len(coherence_scores),
            'cluster_details': cluster_details
        }
    
    # ========================================
    # 3. COVERAGE METRICS
    # ========================================
    
    def evaluate_coverage(
        self, 
        total_reviews: int,
        reviews_with_aspects: int,
        clusters: Dict[int, List[str]] = None
    ) -> Dict[str, Any]:
        """
        Đánh giá độ bao phủ của model.
        
        Metrics:
        - Aspect Coverage: % reviews được gán ít nhất 1 aspect
        - Cluster Balance: Phân bố đều giữa các clusters
        
        Args:
            total_reviews: Tổng số reviews
            reviews_with_aspects: Số reviews có aspect
            clusters: Dict mapping cluster_id -> list of reviews
            
        Returns:
            Dictionary với coverage metrics
        """
        # Coverage rate
        coverage_rate = reviews_with_aspects / total_reviews * 100 if total_reviews > 0 else 0
        
        result = {
            'total_reviews': total_reviews,
            'reviews_with_aspects': reviews_with_aspects,
            'coverage_rate': round(coverage_rate, 2),
            'coverage_interpretation': 'Tốt' if coverage_rate >= 70 else 'Trung bình' if coverage_rate >= 40 else 'Yếu'
        }
        
        # Cluster balance
        if clusters:
            cluster_sizes = [len(reviews) for reviews in clusters.values()]
            if cluster_sizes:
                # Coefficient of variation (CV) - thấp hơn = balanced hơn
                cv = np.std(cluster_sizes) / np.mean(cluster_sizes) if np.mean(cluster_sizes) > 0 else 0
                
                result['cluster_balance'] = {
                    'min_size': min(cluster_sizes),
                    'max_size': max(cluster_sizes),
                    'mean_size': round(np.mean(cluster_sizes), 1),
                    'std_size': round(np.std(cluster_sizes), 1),
                    'coefficient_of_variation': round(cv, 3),
                    'balance_interpretation': 'Tốt' if cv < 0.5 else 'Trung bình' if cv < 1.0 else 'Không cân bằng'
                }
        
        return result
    
    # ========================================
    # 4. SUMMARY QUALITY METRICS
    # ========================================
    
    def evaluate_summary_quality(
        self,
        summary: str,
        source_reviews: List[str],
        aspect_name: str = None
    ) -> Dict[str, Any]:
        """
        Đánh giá chất lượng tóm tắt.
        
        Metrics:
        - Relevance: Similarity giữa summary và source reviews
        - Aspect Focus: Similarity giữa summary và aspect name
        - Length Ratio: Tỷ lệ nén
        
        Args:
            summary: Bản tóm tắt
            source_reviews: Các reviews gốc
            aspect_name: Tên aspect (optional)
            
        Returns:
            Dictionary với summary quality metrics
        """
        if not self._ensure_embedding_model():
            return {'error': 'Không có embedding model'}
        
        # Combine source reviews
        combined_source = ' '.join(source_reviews[:20])[:2000]  # Limit length
        
        # Get embeddings
        texts = [summary, combined_source]
        if aspect_name:
            texts.append(aspect_name)
        
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Relevance: summary vs source
        relevance = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # Aspect focus: summary vs aspect name
        aspect_focus = None
        if aspect_name:
            aspect_focus = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]
        
        # Length ratio
        summary_words = len(summary.split())
        source_words = sum(len(r.split()) for r in source_reviews[:20])
        compression_ratio = summary_words / source_words if source_words > 0 else 0
        
        # Interpretation
        if relevance >= 0.7:
            relevance_interp = "Rất tốt - Summary phản ánh đúng nội dung"
        elif relevance >= 0.5:
            relevance_interp = "Tốt - Summary liên quan"
        elif relevance >= 0.3:
            relevance_interp = "Trung bình - Summary hơi lệch topic"
        else:
            relevance_interp = "Yếu - Summary không liên quan"
        
        result = {
            'relevance_score': round(float(relevance), 4),
            'relevance_interpretation': relevance_interp,
            'summary_words': summary_words,
            'source_words': source_words,
            'compression_ratio': round(compression_ratio, 4)
        }
        
        if aspect_focus is not None:
            result['aspect_focus_score'] = round(float(aspect_focus), 4)
            result['aspect_focus_interpretation'] = 'Tốt' if aspect_focus >= 0.4 else 'Trung bình' if aspect_focus >= 0.2 else 'Yếu'
        
        return result
    
    # ========================================
    # COMPREHENSIVE EVALUATION
    # ========================================
    
    def run_full_evaluation(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        clusters: Dict[int, List[str]],
        summaries: Dict[int, str] = None,
        total_reviews: int = None
    ) -> Dict[str, Any]:
        """
        Chạy đánh giá toàn diện.
        
        Args:
            embeddings: Ma trận embeddings
            labels: Cluster labels
            clusters: Dict mapping cluster_id -> reviews
            summaries: Dict mapping cluster_id -> summary (optional)
            total_reviews: Tổng số reviews (optional)
            
        Returns:
            Comprehensive evaluation report
        """
        logger.info("Đang chạy đánh giá toàn diện...")
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # 1. Clustering quality
        logger.info("1. Đánh giá Clustering Quality...")
        report['clustering_quality'] = self.evaluate_clustering(embeddings, labels)
        
        # 2. Topic coherence
        logger.info("2. Đánh giá Topic Coherence...")
        report['topic_coherence'] = self.evaluate_topic_coherence(clusters)
        
        # 3. Coverage
        if total_reviews:
            logger.info("3. Đánh giá Coverage...")
            reviews_with_aspects = sum(len(r) for r in clusters.values())
            report['coverage'] = self.evaluate_coverage(total_reviews, reviews_with_aspects, clusters)
        
        # 4. Summary quality
        if summaries:
            logger.info("4. Đánh giá Summary Quality...")
            summary_scores = []
            for cluster_id, summary in summaries.items():
                if cluster_id in clusters:
                    score = self.evaluate_summary_quality(summary, clusters[cluster_id])
                    summary_scores.append(score.get('relevance_score', 0))
            
            report['summary_quality'] = {
                'avg_relevance': round(np.mean(summary_scores), 4) if summary_scores else 0,
                'n_summaries_evaluated': len(summary_scores)
            }
        
        # Overall score
        report['overall_assessment'] = self._calculate_overall_score(report)
        
        return report
    
    def _calculate_overall_score(self, report: Dict) -> Dict[str, Any]:
        """Tính điểm tổng hợp."""
        scores = []
        
        # Silhouette: normalize to 0-1 scale
        silhouette = report.get('clustering_quality', {}).get('silhouette_score', 0)
        scores.append((silhouette + 1) / 2)  # [-1,1] -> [0,1]
        
        # Coherence: already 0-1
        coherence = report.get('topic_coherence', {}).get('overall_coherence', 0)
        scores.append(coherence)
        
        # Coverage: percentage to 0-1
        coverage = report.get('coverage', {}).get('coverage_rate', 0)
        scores.append(coverage / 100)
        
        # Summary relevance
        relevance = report.get('summary_quality', {}).get('avg_relevance', 0)
        if relevance > 0:
            scores.append(relevance)
        
        overall = np.mean(scores) if scores else 0
        
        if overall >= 0.7:
            grade = "A - Xuất sắc"
        elif overall >= 0.55:
            grade = "B - Tốt"
        elif overall >= 0.4:
            grade = "C - Trung bình"
        elif overall >= 0.25:
            grade = "D - Cần cải thiện"
        else:
            grade = "F - Yếu"
        
        return {
            'overall_score': round(overall, 4),
            'grade': grade,
            'components': {
                'clustering': round((silhouette + 1) / 2, 4),
                'coherence': round(coherence, 4),
                'coverage': round(coverage / 100, 4) if coverage else 0
            }
        }
    
    def generate_report_markdown(self, evaluation: Dict) -> str:
        """Tạo báo cáo đánh giá dạng Markdown."""
        lines = [
            "# BÁO CÁO ĐÁNH GIÁ MÔ HÌNH",
            "",
            f"**Thời gian:** {evaluation.get('timestamp', 'N/A')}",
            "",
            "---",
            "",
            "## 1. Điểm Tổng Hợp",
            ""
        ]
        
        overall = evaluation.get('overall_assessment', {})
        lines.extend([
            f"**Overall Score:** {overall.get('overall_score', 0):.2%}",
            f"**Grade:** {overall.get('grade', 'N/A')}",
            "",
            "| Thành phần | Điểm |",
            "|------------|------|",
        ])
        
        components = overall.get('components', {})
        for name, score in components.items():
            lines.append(f"| {name.capitalize()} | {score:.2%} |")
        
        lines.extend([
            "",
            "---",
            "",
            "## 2. Clustering Quality",
            ""
        ])
        
        cluster = evaluation.get('clustering_quality', {})
        lines.extend([
            f"- **Silhouette Score:** {cluster.get('silhouette_score', 0)} ({cluster.get('silhouette_interpretation', 'N/A')})",
            f"- **Calinski-Harabasz:** {cluster.get('calinski_harabasz', 0)}",
            f"- **Davies-Bouldin:** {cluster.get('davies_bouldin', 0)} ({cluster.get('davies_interpretation', 'N/A')})",
            f"- **Số clusters:** {cluster.get('n_clusters', 0)}",
            "",
            "---",
            "",
            "## 3. Topic Coherence",
            ""
        ])
        
        coherence = evaluation.get('topic_coherence', {})
        lines.extend([
            f"- **Overall Coherence:** {coherence.get('overall_coherence', 0):.4f}",
            f"- **Interpretation:** {coherence.get('interpretation', 'N/A')}",
            "",
            "---",
            "",
            "## 4. Coverage",
            ""
        ])
        
        coverage = evaluation.get('coverage', {})
        lines.extend([
            f"- **Tổng reviews:** {coverage.get('total_reviews', 0):,}",
            f"- **Reviews có aspect:** {coverage.get('reviews_with_aspects', 0):,}",
            f"- **Coverage Rate:** {coverage.get('coverage_rate', 0):.1f}%",
            ""
        ])
        
        return '\n'.join(lines)


def evaluate_aspect_summarizer(summarizer, category: str, n_aspects: int = 3) -> Dict:
    """
    Hàm tiện ích để đánh giá AspectSummarizer.
    
    Args:
        summarizer: EmbeddingAspectSummarizer instance
        category: Category để test
        n_aspects: Số aspects
        
    Returns:
        Evaluation report
    """
    logger.info(f"Đang đánh giá với category={category}, n_aspects={n_aspects}")
    
    # Run analysis
    result = summarizer.analyze_by_num_aspects(
        n_aspects=n_aspects,
        category=category,
        max_reviews=300
    )
    
    if not result.get('success'):
        return {'error': result.get('error', 'Analysis failed')}
    
    # Prepare data for evaluation
    evaluator = AspectModelEvaluator(summarizer.embedding_model)
    
    # Get embeddings and labels
    filtered_df = summarizer._get_reviews_for_product(category=category, max_reviews=300)
    reviews = filtered_df['review'].tolist()
    
    embeddings = summarizer._create_embeddings(reviews)
    reduced = summarizer._reduce_dimensions(embeddings)
    labels = summarizer._cluster_embeddings(reduced, n_aspects)
    
    # Build clusters dict
    clusters = {}
    for aspect in result['aspects']:
        cluster_id = aspect['aspect_id'] - 1
        clusters[cluster_id] = aspect.get('sample_reviews', [])
    
    # Build summaries dict
    summaries = {
        aspect['aspect_id'] - 1: aspect['summary']
        for aspect in result['aspects']
    }
    
    # Run evaluation
    evaluation = evaluator.run_full_evaluation(
        embeddings=reduced,
        labels=labels,
        clusters=clusters,
        summaries=summaries,
        total_reviews=len(reviews)
    )
    
    # Generate report
    evaluation['markdown_report'] = evaluator.generate_report_markdown(evaluation)
    
    return evaluation
