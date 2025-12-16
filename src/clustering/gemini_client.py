"""
Gemini API client for product clustering and inference.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Dict, Any
from functools import lru_cache

import google.generativeai as genai

from src.config.settings import settings
from src.utils.helpers import retry_with_backoff

logger = logging.getLogger(__name__)


class GeminiClient:
    """
    Client for Google Gemini API.
    
    Handles:
    - API authentication
    - Text generation
    - Batch processing for clustering
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Gemini API key (or from environment)
        """
        self.api_key = api_key or settings.gemini.api_key or os.getenv("GEMINI_API_KEY")
        self.config = settings.gemini
        self._model = None
        
        if not self.api_key:
            logger.warning("Gemini API key not provided. Set GEMINI_API_KEY environment variable.")
        else:
            self._setup_client()
            
    def _setup_client(self) -> None:
        """Configure Gemini API client."""
        try:
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.config.model_name)
            logger.info(f"Gemini client initialized with model: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise
            
    @property
    def is_available(self) -> bool:
        """Check if Gemini client is ready."""
        return self._model is not None
        
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate text response from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        if not self.is_available:
            raise RuntimeError("Gemini client not initialized")
            
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature or self.config.temperature
        )
        
        try:
            response = self._model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if response.text:
                return response.text.strip()
            else:
                logger.warning("Empty response from Gemini")
                return ""
                
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise
            
    def categorize_products(
        self,
        product_infos: List[Dict[str, str]],
        existing_categories: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        Categorize products into groups using Gemini.
        
        Args:
            product_infos: List of dicts with 'title' and optionally 'url', 'review'
            existing_categories: Optional list of existing category names to use
            
        Returns:
            List of dicts with 'title' and 'category'
        """
        if not self.is_available:
            logger.warning("Gemini not available, returning empty categories")
            return [{'title': p.get('title', ''), 'category': 'Unknown'} for p in product_infos]
            
        logger.info(f"Categorizing {len(product_infos)} products...")
        
        results = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(product_infos), batch_size):
            batch = product_infos[i:i + batch_size]
            batch_results = self._categorize_batch(batch, existing_categories)
            results.extend(batch_results)
            
            logger.info(f"Categorized {min(i + batch_size, len(product_infos))}/{len(product_infos)} products")
            
        return results
        
    def _categorize_batch(
        self,
        products: List[Dict[str, str]],
        existing_categories: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        Categorize a batch of products.
        
        Args:
            products: List of product info dicts
            existing_categories: Optional existing categories
            
        Returns:
            List of dicts with category assigned
        """
        # Build product list for prompt
        product_list = []
        for i, p in enumerate(products):
            title = p.get('title', 'Unknown')
            product_list.append(f"{i + 1}. {title}")
            
        products_text = "\n".join(product_list)
        
        # Build category guidance
        category_guidance = ""
        if existing_categories:
            category_guidance = (
                f"\nUse these existing categories when appropriate: "
                f"{', '.join(existing_categories)}\n"
            )
            
        prompt = f"""Categorize each product into a descriptive category. Return ONLY a numbered list with format:
1. [Category Name]
2. [Category Name]
...

Categories should be descriptive (e.g., "Electronics - Headphones", "Home & Kitchen - Cookware", "Toys & Games - Action Figures").
{category_guidance}
Products:
{products_text}

Categories:"""

        try:
            response = self.generate(prompt, max_tokens=500)
            categories = self._parse_category_response(response, len(products))
            
            return [
                {'title': p.get('title', ''), 'category': cat}
                for p, cat in zip(products, categories)
            ]
            
        except Exception as e:
            logger.error(f"Batch categorization failed: {e}")
            return [{'title': p.get('title', ''), 'category': 'Unknown'} for p in products]
            
    def _parse_category_response(self, response: str, expected_count: int) -> List[str]:
        """
        Parse category response from Gemini.
        
        Args:
            response: Raw response text
            expected_count: Expected number of categories
            
        Returns:
            List of category names
        """
        import re
        
        categories = []
        lines = response.strip().split('\n')
        
        for line in lines:
            # Match patterns like "1. Category Name" or "1) Category Name"
            match = re.match(r'^\d+[\.\)]\s*(.+)$', line.strip())
            if match:
                category = match.group(1).strip()
                # Remove any leading/trailing brackets
                category = re.sub(r'^\[|\]$', '', category)
                categories.append(category)
                
        # Pad with "Unknown" if not enough categories
        while len(categories) < expected_count:
            categories.append("Unknown")
            
        return categories[:expected_count]
        
    def extract_aspects_from_review(self, review: str) -> Dict[str, str]:
        """
        Extract aspects and sentiments from a review.
        
        Args:
            review: Review text
            
        Returns:
            Dict mapping aspect to sentiment
        """
        if not self.is_available or not review:
            return {}
            
        prompt = f"""Analyze this product review and extract aspects with their sentiment.
Return in format:
aspect1: positive/negative/neutral
aspect2: positive/negative/neutral

Common aspects: quality, price, shipping, packaging, durability, ease_of_use, value, appearance, customer_service

Review: {review[:500]}

Aspects:"""

        try:
            response = self.generate(prompt, max_tokens=200)
            return self._parse_aspects_response(response)
        except Exception as e:
            logger.debug(f"Aspect extraction failed: {e}")
            return {}
            
    def _parse_aspects_response(self, response: str) -> Dict[str, str]:
        """Parse aspects from Gemini response."""
        import re
        
        aspects = {}
        lines = response.strip().split('\n')
        
        for line in lines:
            match = re.match(r'^(\w+)\s*:\s*(positive|negative|neutral)', line.strip(), re.IGNORECASE)
            if match:
                aspect = match.group(1).lower()
                sentiment = match.group(2).lower()
                aspects[aspect] = sentiment
                
        return aspects
            aspects[aspect] = sentiment
                
        return aspects

    def infer_product_info_batch(
        self,
        reviews: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Infer product title and rating from reviews in batch.
        
        Args:
            reviews: List of review texts
            
        Returns:
            List of dicts with 'title' and 'rating'
        """
        if not self.is_available:
            return [{'title': None, 'rating': None} for _ in reviews]
            
        logger.info(f"Inferring info for {len(reviews)} reviews...")
        
        results = []
        batch_size = 10  # Smaller batch size for complex inference
        
        from src.utils.helpers import clean_text
        
        for i in range(0, len(reviews), batch_size):
            batch_reviews = reviews[i:i + batch_size]
            batch_results = self._infer_info_batch_process(batch_reviews)
            results.extend(batch_results)
            
        return results

    def _infer_info_batch_process(self, reviews: List[str]) -> List[Dict[str, Any]]:
        """Process a single batch for inference."""
        reviews_formatted = []
        for idx, rev in enumerate(reviews):
            # Truncate to save tokens but keep enough context
            clean_rev = rev[:500].replace('\n', ' ')
            reviews_formatted.append(f"Review {idx+1}: {clean_rev}")
            
        reviews_block = "\n".join(reviews_formatted)
        
        prompt = f"""Analyze the following product reviews. For each review, infer the likely Product Name (Title) and a Rating (1-5 stars) based on the sentiment.

Reviews:
{reviews_block}

Return a valid JSON list of objects. Each object must have "title" (string) and "rating" (integer 1-5).
If title cannot be inferred, use "Generic Product".
If rating cannot be inferred, use 3.

Format:
[
  {{"title": "Product A", "rating": 5}},
  {{"title": "Product B", "rating": 1}}
]
"""
        
        try:
            response = self.generate(prompt, max_tokens=1000)
            import json
            import re
            
            # Extract JSON block
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                json_str = match.group(0)
                data = json.loads(json_str)
                
                # Validate length
                if len(data) != len(reviews):
                    logger.warning(f"Inference count mismatch: got {len(data)}, expected {len(reviews)}")
                    # Pad or truncate
                    if len(data) < len(reviews):
                        data.extend([{'title': 'Unknown', 'rating': 3}] * (len(reviews) - len(data)))
                    else:
                        data = data[:len(reviews)]
                return data
            else:
                logger.warning("Could not parse JSON from Gemini response")
                return [{'title': None, 'rating': None} for _ in reviews]
                
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            return [{'title': None, 'rating': None} for _ in reviews]
