"""
Configuration settings for Walmart Product Review Analysis Pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Load .env file (optional - for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required, use environment variables directly

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class KaggleConfig:
    """Kaggle dataset configuration."""
    dataset_name: str = "promptcloud/walmart-product-reviews-dataset"
    file_name: str = "marketing_sample_for_walmart_com-walmart_product_reviews__20200401_20200630__30k_data.csv"


@dataclass
class DateConfig:
    """Date shifting configuration."""
    years_to_shift: int = 10  # Số năm lùi về quá khứ


@dataclass
class ScraperConfig:
    """Web scraper configuration."""
    headless: bool = True
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 2.0  # seconds
    rate_limit_delay: float = 1.5  # seconds between requests
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )


@dataclass
class GeminiConfig:
    """Google Gemini API configuration."""
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    model_name: str = "gemini-2.0-flash"  # Updated model name
    max_tokens: int = 2048
    temperature: float = 0.3
    batch_size: int = 20  # Products per API call for clustering


@dataclass
class AnalysisConfig:
    """Sentiment analysis configuration."""
    # Common aspects to analyze
    aspects: List[str] = field(default_factory=lambda: [
        "quality",      # Chất lượng sản phẩm
        "price",        # Giá cả
        "shipping",     # Giao hàng
        "packaging",    # Đóng gói
        "durability",   # Độ bền
        "ease_of_use",  # Dễ sử dụng
        "value",        # Giá trị so với tiền
        "appearance",   # Ngoại hình
        "customer_service",  # CSKH
    ])
    
    # Sentiment keywords (Vietnamese + English)
    positive_keywords: List[str] = field(default_factory=lambda: [
        "great", "excellent", "amazing", "love", "perfect", "best",
        "good", "wonderful", "fantastic", "recommend", "satisfied",
        "happy", "awesome", "outstanding", "superb", "quality"
    ])
    
    negative_keywords: List[str] = field(default_factory=lambda: [
        "bad", "terrible", "horrible", "hate", "worst", "poor",
        "disappointed", "broken", "defective", "waste", "awful",
        "useless", "junk", "cheap", "garbage", "fail"
    ])


@dataclass 
class ColumnMapping:
    """Column name mappings and metadata."""
    # Original -> normalized column names
    column_renames: Dict[str, str] = field(default_factory=lambda: {
        "uniq _id": "uniq_id",
        "crawl _timestamp": "crawl_timestamp",
        "reviewer _name": "reviewer_name",
        "review _upvotes": "review_upvotes",
        "review _downvotes": "review_downvotes",
        "verified _purchaser": "verified_purchaser",
        "recommended _purchase": "recommended_purchase",
        "review _date": "review_date",
        "five _star": "five_star",
        "four _star": "four_star",
        "three _star": "three_star",
        "two _star": "two_star",
        "one _star": "one_star",
    })
    
    # Columns that can be scraped from PageURL
    scrapable_columns: List[str] = field(default_factory=lambda: [
        "title", "rating", "review"
    ])
    
    # Date columns to shift
    date_columns: List[str] = field(default_factory=lambda: [
        "crawl_timestamp", "review_date"
    ])
    
    # Required columns
    required_columns: List[str] = field(default_factory=lambda: [
        "uniq_id", "pageurl", "rating"
    ])


@dataclass
class Settings:
    """Main settings container."""
    kaggle: KaggleConfig = field(default_factory=KaggleConfig)
    date: DateConfig = field(default_factory=DateConfig)
    scraper: ScraperConfig = field(default_factory=ScraperConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    columns: ColumnMapping = field(default_factory=ColumnMapping)
    
    # File paths
    raw_data_path: Path = DATA_DIR / "raw_data.csv"
    processed_data_path: Path = DATA_DIR / "processed_data.csv"
    clustered_data_path: Path = OUTPUT_DIR / "clustered_products.csv"
    sentiment_data_path: Path = OUTPUT_DIR / "sentiment_analysis.csv"
    report_path: Path = OUTPUT_DIR / "analysis_report.md"


# Global settings instance
settings = Settings()
