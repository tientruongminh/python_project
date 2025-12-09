"""
RAG Pipeline cho Aspect-Based Summarization.

Components:
1. AspectVectorStore: Lưu trữ pre-computed aspects với embeddings
2. AspectRetriever: Retrieve relevant aspects từ query
3. RAGQueryEngine: Query với LLM augmentation
4. PreComputePipeline: Batch process all categories
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Storage paths
VECTOR_STORE_DIR = Path(__file__).parent.parent.parent / "outputs" / "vector_store"
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AspectDocument:
    """Một document trong vector store."""
    doc_id: str
    category: str
    aspect_name: str
    summary: str
    sentiment: Dict[str, Any]
    review_count: int
    sample_reviews: List[str]
    embedding: Optional[np.ndarray] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'doc_id': self.doc_id,
            'category': self.category,
            'aspect_name': self.aspect_name,
            'summary': self.summary,
            'sentiment': self.sentiment,
            'review_count': self.review_count,
            'sample_reviews': self.sample_reviews,
            'created_at': self.created_at
        }


class AspectVectorStore:
    """
    Vector store để lưu trữ pre-computed aspects.
    
    Sử dụng simple in-memory store với cosine similarity.
    Có thể upgrade lên FAISS nếu cần scale.
    """
    
    def __init__(self, store_path: Optional[Path] = None):
        """
        Khởi tạo vector store.
        
        Args:
            store_path: Path to store files
        """
        self.store_path = store_path or VECTOR_STORE_DIR
        self.documents: List[AspectDocument] = []
        self.embeddings: Optional[np.ndarray] = None
        self.embedding_model = None
        
        self._init_embedding_model()
        self._load_store()
    
    def _init_embedding_model(self):
        """Load embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Đã tải embedding model cho vector store")
        except ImportError:
            logger.error("Cần cài đặt: pip install sentence-transformers")
    
    def _load_store(self):
        """Load từ disk nếu có."""
        docs_path = self.store_path / "aspects_docs.pkl"
        embeddings_path = self.store_path / "aspects_embeddings.npy"
        
        if docs_path.exists() and embeddings_path.exists():
            try:
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                self.embeddings = np.load(embeddings_path)
                logger.info(f"Đã load {len(self.documents)} documents từ store")
            except Exception as e:
                logger.warning(f"Không thể load store: {e}")
    
    def save_store(self):
        """Save store vào disk."""
        docs_path = self.store_path / "aspects_docs.pkl"
        embeddings_path = self.store_path / "aspects_embeddings.npy"
        
        try:
            with open(docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
            if self.embeddings is not None:
                np.save(embeddings_path, self.embeddings)
            logger.info(f"Đã save {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Không thể save store: {e}")
    
    def add_document(self, doc: AspectDocument):
        """
        Thêm document vào store.
        
        Args:
            doc: AspectDocument to add
        """
        # Create embedding if not exists
        if doc.embedding is None and self.embedding_model:
            text = f"{doc.category} {doc.aspect_name} {doc.summary}"
            doc.embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        
        self.documents.append(doc)
        
        # Update embeddings matrix
        if doc.embedding is not None:
            if self.embeddings is None:
                self.embeddings = doc.embedding.reshape(1, -1)
            else:
                self.embeddings = np.vstack([self.embeddings, doc.embedding])
    
    def add_documents(self, docs: List[AspectDocument]):
        """Add nhiều documents."""
        for doc in docs:
            self.add_document(doc)
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        category_filter: Optional[str] = None
    ) -> List[Tuple[AspectDocument, float]]:
        """
        Tìm kiếm documents tương đồng với query.
        
        Args:
            query: Query text
            top_k: Số kết quả trả về
            category_filter: Lọc theo category (optional)
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if not self.documents or self.embedding_model is None:
            return []
        
        # Embed query
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        
        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get results with filter
        results = []
        for idx, (doc, sim) in enumerate(zip(self.documents, similarities)):
            if category_filter and doc.category.lower() != category_filter.lower():
                continue
            results.append((doc, float(sim)))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def get_categories(self) -> List[str]:
        """Get danh sách categories trong store."""
        return list(set(doc.category for doc in self.documents))
    
    def get_documents_by_category(self, category: str) -> List[AspectDocument]:
        """Get tất cả documents của một category."""
        return [doc for doc in self.documents if doc.category.lower() == category.lower()]
    
    def clear(self):
        """Xóa tất cả documents."""
        self.documents = []
        self.embeddings = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get thống kê về store."""
        if not self.documents:
            return {'status': 'empty', 'n_documents': 0}
        
        categories = self.get_categories()
        return {
            'status': 'ready',
            'n_documents': len(self.documents),
            'n_categories': len(categories),
            'categories': categories,
            'embedding_dim': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'last_updated': max(doc.created_at for doc in self.documents) if self.documents else None
        }


class RAGQueryEngine:
    """
    Query engine với RAG (Retrieval-Augmented Generation).
    
    Sử dụng vector store để retrieve relevant aspects,
    sau đó dùng LLM để generate response.
    """
    
    def __init__(self, vector_store: AspectVectorStore):
        """
        Khởi tạo query engine.
        
        Args:
            vector_store: AspectVectorStore instance
        """
        self.vector_store = vector_store
        self.gemini_client = None
        self._init_gemini()
    
    def _init_gemini(self):
        """Initialize Gemini client."""
        try:
            from src.clustering.gemini_client import GeminiClient
            self.gemini_client = GeminiClient()
            if not self.gemini_client.is_available:
                logger.warning("Gemini API không khả dụng - sẽ trả về raw results")
                self.gemini_client = None
        except Exception as e:
            logger.warning(f"Không thể khởi tạo Gemini: {e}")
    
    def query(
        self, 
        question: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Query với RAG.
        
        Args:
            question: User's question
            top_k: Số aspects liên quan để retrieve
            category_filter: Lọc theo category
            use_llm: Có sử dụng LLM để generate response không
            
        Returns:
            Response dictionary
        """
        # Step 1: Retrieve relevant aspects
        results = self.vector_store.search(question, top_k=top_k, category_filter=category_filter)
        
        if not results:
            return {
                'success': False,
                'error': 'Không tìm thấy aspects liên quan. Hãy chạy pre-compute trước.',
                'query': question
            }
        
        # Step 2: Format context
        context_parts = []
        retrieved_aspects = []
        
        for doc, similarity in results:
            context_parts.append(f"""
Category: {doc.category}
Aspect: {doc.aspect_name}
Summary: {doc.summary}
Sentiment: {doc.sentiment.get('interpretation', 'N/A')} ({doc.sentiment.get('positive_pct', 0)}% positive, {doc.sentiment.get('negative_pct', 0)}% negative)
Review Count: {doc.review_count}
""")
            retrieved_aspects.append({
                'category': doc.category,
                'aspect_name': doc.aspect_name,
                'summary': doc.summary,
                'sentiment': doc.sentiment,
                'review_count': doc.review_count,
                'similarity': round(similarity, 4)
            })
        
        context = "\n---\n".join(context_parts)
        
        # Step 3: Generate response with LLM (optional)
        if use_llm and self.gemini_client:
            response = self._generate_llm_response(question, context)
        else:
            response = self._format_raw_response(question, retrieved_aspects)
        
        return {
            'success': True,
            'query': question,
            'response': response,
            'retrieved_aspects': retrieved_aspects,
            'n_results': len(results)
        }
    
    def _generate_llm_response(self, question: str, context: str) -> str:
        """Generate response using LLM."""
        prompt = f"""Dựa trên dữ liệu phân tích đánh giá sản phẩm sau, hãy trả lời câu hỏi của người dùng:

**Dữ liệu phân tích:**
{context}

**Câu hỏi:**
{question}

**Hướng dẫn:**
- Trả lời bằng tiếng Việt
- Sử dụng ngôn ngữ tự nhiên, dễ hiểu
- Đề cập số liệu cụ thể từ dữ liệu
- Nếu câu hỏi không liên quan đến dữ liệu, nói rõ

**Trả lời:**"""
        
        try:
            response = self.gemini_client.generate(prompt, max_tokens=500)
            return response.strip() if response else "Không thể generate response."
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return f"Lỗi khi generate response: {e}"
    
    def _format_raw_response(self, question: str, aspects: List[Dict]) -> str:
        """Format raw response without LLM."""
        lines = [f"Kết quả cho câu hỏi: '{question}'\n"]
        
        for i, asp in enumerate(aspects, 1):
            lines.append(f"{i}. **{asp['aspect_name']}** ({asp['category']})")
            lines.append(f"   - {asp['summary']}")
            lines.append(f"   - Sentiment: {asp['sentiment'].get('interpretation', 'N/A')}")
            lines.append(f"   - {asp['review_count']} reviews")
            lines.append("")
        
        return "\n".join(lines)


class PreComputePipeline:
    """
    Pipeline để pre-compute tất cả aspects cho mọi category.
    """
    
    def __init__(self, df: pd.DataFrame, vector_store: AspectVectorStore):
        """
        Khởi tạo pipeline.
        
        Args:
            df: DataFrame với reviews
            vector_store: AspectVectorStore để lưu kết quả
        """
        self.df = df
        self.vector_store = vector_store
        self.summarizer = None
    
    def _init_summarizer(self):
        """Initialize EmbeddingAspectSummarizer."""
        try:
            from src.analysis.aspect_summarizer import EmbeddingAspectSummarizer
            self.summarizer = EmbeddingAspectSummarizer(self.df, fast_mode=True, use_cache=True)
        except Exception as e:
            logger.error(f"Không thể khởi tạo summarizer: {e}")
    
    def get_categories(self, min_reviews: int = 50) -> List[str]:
        """Get danh sách categories có đủ reviews."""
        if 'product_category' not in self.df.columns:
            return []
        
        category_counts = self.df['product_category'].value_counts()
        return [cat for cat, count in category_counts.items() 
                if count >= min_reviews and cat not in ['Unknown', 'Other']]
    
    def process_category(
        self, 
        category: str, 
        n_aspects: int = 5,
        max_reviews: int = 300
    ) -> List[AspectDocument]:
        """
        Process một category và tạo AspectDocuments.
        
        Args:
            category: Category name
            n_aspects: Số aspects để discover
            max_reviews: Max reviews per category
            
        Returns:
            List of AspectDocuments
        """
        if self.summarizer is None:
            self._init_summarizer()
        
        if self.summarizer is None:
            return []
        
        logger.info(f"Processing category: {category}")
        
        # Analyze
        result = self.summarizer.analyze_by_num_aspects(
            n_aspects=n_aspects,
            category=category,
            max_reviews=max_reviews
        )
        
        if not result.get('success'):
            logger.warning(f"Failed to analyze {category}: {result.get('error')}")
            return []
        
        # Create documents
        documents = []
        for aspect in result.get('aspects', []):
            doc = AspectDocument(
                doc_id=f"{category}_{aspect['aspect_id']}",
                category=category,
                aspect_name=aspect['aspect_name'],
                summary=aspect['summary'],
                sentiment=aspect.get('sentiment', {}),
                review_count=aspect['review_count'],
                sample_reviews=aspect.get('sample_reviews', [])[:3]
            )
            documents.append(doc)
        
        return documents
    
    def run_full_pipeline(
        self, 
        n_aspects: int = 5,
        max_reviews_per_category: int = 300,
        max_categories: Optional[int] = None,
        progress_callback = None
    ) -> Dict[str, Any]:
        """
        Chạy pipeline cho tất cả categories.
        
        Args:
            n_aspects: Số aspects per category
            max_reviews_per_category: Max reviews per category
            max_categories: Limit số categories (for testing)
            progress_callback: Callback function(current, total, category)
            
        Returns:
            Pipeline results
        """
        # Clear existing store
        self.vector_store.clear()
        
        # Get categories
        categories = self.get_categories()
        if max_categories:
            categories = categories[:max_categories]
        
        if not categories:
            return {'success': False, 'error': 'Không tìm thấy categories'}
        
        logger.info(f"Bắt đầu pre-compute cho {len(categories)} categories")
        
        results = {
            'processed_categories': [],
            'failed_categories': [],
            'total_documents': 0
        }
        
        for i, category in enumerate(categories):
            if progress_callback:
                progress_callback(i + 1, len(categories), category)
            
            try:
                documents = self.process_category(
                    category, 
                    n_aspects=n_aspects,
                    max_reviews=max_reviews_per_category
                )
                
                if documents:
                    self.vector_store.add_documents(documents)
                    results['processed_categories'].append({
                        'category': category,
                        'n_aspects': len(documents)
                    })
                    results['total_documents'] += len(documents)
                else:
                    results['failed_categories'].append(category)
                    
            except Exception as e:
                logger.error(f"Error processing {category}: {e}")
                results['failed_categories'].append(category)
        
        # Save store
        self.vector_store.save_store()
        
        results['success'] = True
        results['n_processed'] = len(results['processed_categories'])
        results['n_failed'] = len(results['failed_categories'])
        
        return results


def create_rag_pipeline(df: pd.DataFrame) -> Tuple[AspectVectorStore, RAGQueryEngine, PreComputePipeline]:
    """
    Factory function để tạo RAG pipeline components.
    
    Args:
        df: DataFrame với reviews
        
    Returns:
        Tuple of (vector_store, query_engine, precompute_pipeline)
    """
    vector_store = AspectVectorStore()
    query_engine = RAGQueryEngine(vector_store)
    precompute_pipeline = PreComputePipeline(df, vector_store)
    
    return vector_store, query_engine, precompute_pipeline
