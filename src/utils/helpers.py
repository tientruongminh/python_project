"""
Utility helper functions.
"""
from __future__ import annotations

import re
import logging
from typing import Optional
from functools import wraps
from time import sleep

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def to_snake_case(name: str) -> str:
    """
    Convert a column name to snake_case.
    
    Args:
        name: Original column name (CamelCase, PascalCase, or with spaces)
        
    Returns:
        Normalized snake_case name
        
    Example:
        >>> to_snake_case("ReviewUpvotes")
        'review_upvotes'
        >>> to_snake_case("five Star")
        'five_star'
    """
    # Handle non-breaking spaces
    name = name.replace("\xa0", " ")
    name = name.strip()
    
    # Split CamelCase / PascalCase
    name = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    
    # Replace non-alphanumeric with underscore
    name = re.sub(r'[^a-zA-Z0-9]+', '_', name)
    
    # Lowercase
    name = name.lower()
    
    # Remove duplicate underscores
    name = re.sub(r'_+', '_', name).strip('_')
    
    return name


def normalize_column_name(col: str) -> str:
    """
    Normalize column name with specific mappings for known issues.
    
    Args:
        col: Column name to normalize
        
    Returns:
        Normalized column name
    """
    col = str(col)
    col = col.replace("\xa0", " ")
    col = col.strip()
    col = re.sub(r"\s+", " ", col)
    
    # Specific mappings for "x _y" patterns
    mapping = {
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
    }
    
    return mapping.get(col, to_snake_case(col))


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        sleep(delay)
                        delay = min(delay * 2, max_delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}")
                        
            raise last_exception
        return wrapper
    return decorator


def extract_product_id_from_url(url: str) -> Optional[str]:
    """
    Extract product ID from Walmart URL.
    
    Args:
        url: Walmart product page URL
        
    Returns:
        Product ID if found, None otherwise
        
    Example:
        >>> extract_product_id_from_url("https://www.walmart.com/ip/Product-Name/123456789")
        '123456789'
    """
    if not url:
        return None
        
    # Pattern: /ip/product-name/ID or /ip/ID
    patterns = [
        r'/ip/[^/]+/(\d+)',  # /ip/product-name/123456789
        r'/ip/(\d+)',        # /ip/123456789
        r'/(\d{9,12})(?:\?|$)',  # Any 9-12 digit number at end
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
            
    return None


def clean_text(text: str) -> str:
    """
    Clean review text by removing extra whitespace and special characters.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
        
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def format_number(num: float, decimal_places: int = 2) -> str:
    """
    Format a number for display.
    
    Args:
        num: Number to format
        decimal_places: Number of decimal places
        
    Returns:
        Formatted string
    """
    if num >= 1_000_000:
        return f"{num / 1_000_000:.{decimal_places}f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.{decimal_places}f}K"
    else:
        return f"{num:.{decimal_places}f}"
