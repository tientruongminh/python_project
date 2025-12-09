"""
Gemini URL Context Scraper.

Sử dụng Gemini API với URL Context để scrape thông tin sản phẩm từ Walmart URLs.
Thay thế Selenium-based scraper với approach nhanh và ổn định hơn.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)


@dataclass
class ProductInfo:
    """Thông tin sản phẩm được scrape."""
    title: Optional[str] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None
    price: Optional[str] = None
    five_star_pct: Optional[float] = None
    four_star_pct: Optional[float] = None
    three_star_pct: Optional[float] = None
    two_star_pct: Optional[float] = None
    one_star_pct: Optional[float] = None
    description: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'rating': self.rating,
            'review_count': self.review_count,
            'price': self.price,
            'five_star_pct': self.five_star_pct,
            'four_star_pct': self.four_star_pct,
            'three_star_pct': self.three_star_pct,
            'two_star_pct': self.two_star_pct,
            'one_star_pct': self.one_star_pct,
            'description': self.description,
            'error': self.error
        }


class GeminiURLScraper:
    """
    Scraper sử dụng Gemini URL Context API.
    
    Ưu điểm so với Selenium:
    - Không cần WebDriver
    - Nhanh hơn (không render trang)
    - Ổn định hơn
    
    Lưu ý:
    - Cần API key trong GEMINI_API_KEY environment variable
    - Có giới hạn quota
    """
    
    def __init__(self, model: str = "gemini-2.0-flash"):
        """
        Khởi tạo scraper.
        
        Args:
            model: Gemini model name
        """
        self.model = model
        self.client = None
        self.is_available = False
        self._init_client()
        
        # Rate limiting
        self.requests_per_minute = 60
        self.last_request_time = 0
        self.request_count = 0
        
    def _init_client(self):
        """Initialize Gemini client."""
        try:
            from google import genai
            
            api_key = os.environ.get('GEMINI_API_KEY')
            if not api_key:
                logger.warning("GEMINI_API_KEY không được set")
                return
            
            self.client = genai.Client(api_key=api_key)
            self.is_available = True
            logger.info("Gemini URL Scraper đã khởi tạo thành công")
            
        except ImportError:
            logger.error("Cần cài đặt: pip install google-genai")
        except Exception as e:
            logger.error(f"Lỗi khởi tạo Gemini client: {e}")
    
    def _rate_limit(self):
        """Apply rate limiting."""
        current_time = time.time()
        
        # Reset counter mỗi phút
        if current_time - self.last_request_time > 60:
            self.request_count = 0
            self.last_request_time = current_time
        
        # Wait nếu vượt quota
        if self.request_count >= self.requests_per_minute:
            wait_time = 60 - (current_time - self.last_request_time)
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                self.request_count = 0
                self.last_request_time = time.time()
        
        self.request_count += 1
    
    def scrape_product(self, url: str, max_retries: int = 3) -> ProductInfo:
        """
        Scrape thông tin sản phẩm từ URL.
        
        Args:
            url: Walmart product URL
            max_retries: Số lần retry nếu lỗi
            
        Returns:
            ProductInfo object
        """
        if not self.is_available:
            return ProductInfo(error="Gemini client không khả dụng")
        
        self._rate_limit()
        
        prompt = """Analyze this Walmart product page and extract the following information in JSON format:
{
    "title": "Product title",
    "rating": 4.5,
    "review_count": 1234,
    "price": "$29.99",
    "five_star_pct": 60,
    "four_star_pct": 25,
    "three_star_pct": 8,
    "two_star_pct": 4,
    "one_star_pct": 3,
    "description": "Brief product description"
}

If any field is not available, use null.
Return ONLY the JSON object, no other text."""

        for attempt in range(max_retries):
            try:
                from google.genai.types import GenerateContentConfig
                
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[prompt],
                    config=GenerateContentConfig(
                        tool_config={
                            "url_context": {
                                "urls": [url]
                            }
                        }
                    )
                )
                
                # Parse response
                if response.candidates and response.candidates[0].content.parts:
                    text = response.candidates[0].content.parts[0].text
                    return self._parse_response(text)
                else:
                    return ProductInfo(error="Empty response from Gemini")
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
        return ProductInfo(error=f"Failed after {max_retries} attempts")
    
    def _parse_response(self, text: str) -> ProductInfo:
        """Parse JSON response từ Gemini."""
        try:
            # Clean up response
            text = text.strip()
            if text.startswith('```json'):
                text = text[7:]
            if text.startswith('```'):
                text = text[3:]
            if text.endswith('```'):
                text = text[:-3]
            text = text.strip()
            
            data = json.loads(text)
            
            return ProductInfo(
                title=data.get('title'),
                rating=data.get('rating'),
                review_count=data.get('review_count'),
                price=data.get('price'),
                five_star_pct=data.get('five_star_pct'),
                four_star_pct=data.get('four_star_pct'),
                three_star_pct=data.get('three_star_pct'),
                two_star_pct=data.get('two_star_pct'),
                one_star_pct=data.get('one_star_pct'),
                description=data.get('description')
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return ProductInfo(error=f"JSON parse error: {e}")
    
    def scrape_batch(
        self, 
        urls: List[str], 
        progress_callback=None
    ) -> List[ProductInfo]:
        """
        Scrape nhiều URLs.
        
        Args:
            urls: List of URLs
            progress_callback: Optional callback(current, total, url)
            
        Returns:
            List of ProductInfo objects
        """
        results = []
        total = len(urls)
        
        for i, url in enumerate(urls):
            if progress_callback:
                progress_callback(i + 1, total, url)
            
            result = self.scrape_product(url)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{total} URLs processed")
        
        return results


class GeminiURLScraperV2:
    """
    Alternative implementation using standard Gemini API with URL as context.
    Fallback nếu URL Context tool không hoạt động.
    """
    
    def __init__(self):
        self.gemini_client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize standard Gemini client."""
        try:
            from src.clustering.gemini_client import GeminiClient
            self.gemini_client = GeminiClient()
            if not self.gemini_client.is_available:
                logger.warning("Gemini client không khả dụng")
                self.gemini_client = None
        except Exception as e:
            logger.error(f"Lỗi khởi tạo: {e}")
    
    def extract_product_id(self, url: str) -> Optional[str]:
        """Extract product ID from URL."""
        import re
        match = re.search(r'/product/(\d+)', url)
        return match.group(1) if match else None
    
    def scrape_product(self, url: str) -> ProductInfo:
        """
        Scrape using product ID approach.
        
        Args:
            url: Walmart URL
            
        Returns:
            ProductInfo
        """
        if not self.gemini_client:
            return ProductInfo(error="Gemini client không khả dụng")
        
        product_id = self.extract_product_id(url)
        if not product_id:
            return ProductInfo(error="Không thể extract product ID")
        
        # Use Gemini to infer product info based on URL pattern
        prompt = f"""Based on this Walmart product URL: {url}
        
The product ID is: {product_id}

Since I cannot directly access the URL, please provide a reasonable estimate of what this product might be based on the URL pattern and product ID.

If you have any cached knowledge about Walmart products, try to match this ID.

Return as JSON:
{{
    "title": "Estimated product title or null",
    "category": "Product category if known"
}}"""

        try:
            response = self.gemini_client.generate(prompt, max_tokens=200)
            if response:
                data = json.loads(response.strip())
                return ProductInfo(title=data.get('title'))
        except:
            pass
        
        return ProductInfo(error="Could not extract info")


def test_scraper():
    """Test function cho scraper."""
    scraper = GeminiURLScraper()
    
    test_urls = [
        "https://www.walmart.com/reviews/product/36907838",
        "https://www.walmart.com/reviews/product/708236785",
    ]
    
    for url in test_urls:
        print(f"\nScraping: {url}")
        result = scraper.scrape_product(url)
        print(f"Result: {result.to_dict()}")


if __name__ == "__main__":
    test_scraper()
