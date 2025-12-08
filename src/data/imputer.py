"""
Data imputer module for filling missing values.

Features:
- URL validation (remove dead/invalid URLs)
- Web scraping for missing reviews
- LLM inference for context-based imputation
"""
from __future__ import annotations

import logging
import requests
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from src.config.settings import settings
from src.scrapers.walmart_scraper import WalmartScraper

logger = logging.getLogger(__name__)


class DataImputer:
    """
    Imputer for filling missing values in DataFrame.
    
    Uses:
    - URL validation to remove invalid links
    - Web scraping to fill missing reviews
    - LLM inference for context-based imputation
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize imputer with DataFrame.
        
        Args:
            df: DataFrame with missing values
        """
        self.df = df.copy()
        self.config = settings
        self.scraped_cache: Dict[str, Dict[str, Any]] = {}
        self.invalid_urls: List[str] = []
        
    def impute_all(
        self,
        use_scraping: bool = True,
        use_llm: bool = True,
        validate_urls: bool = True,
        max_scrape_urls: Optional[int] = None,
        remove_invalid: bool = True
    ) -> pd.DataFrame:
        """
        Run all imputation methods.
        
        Args:
            use_scraping: Whether to use web scraping
            use_llm: Whether to use LLM inference
            validate_urls: Whether to validate URLs first
            max_scrape_urls: Maximum URLs to scrape
            remove_invalid: Whether to remove rows with invalid URLs
            
        Returns:
            DataFrame with filled values
        """
        logger.info("Starting data imputation...")
        
        initial_rows = len(self.df)
        missing_before = self.df.isna().sum().sum()
        
        # Step 1: Validate URLs and remove invalid ones
        if validate_urls:
            self.validate_and_clean_urls(remove_invalid=remove_invalid)
        
        # Step 2: Scrape missing data
        if use_scraping:
            self.impute_from_scraping(max_urls=max_scrape_urls)
            
        # Step 3: Use LLM for remaining missing
        if use_llm:
            self.impute_from_llm()
            
        # Step 4: Fill remaining with defaults
        self.fill_defaults()
        
        # Get missing info after
        missing_after = self.df.isna().sum().sum()
        final_rows = len(self.df)
        
        logger.info(f"Imputation complete:")
        logger.info(f"  Rows: {initial_rows} -> {final_rows} (removed {initial_rows - final_rows})")
        logger.info(f"  Missing values: {missing_before} -> {missing_after}")
        
        return self.df
    
    # ================================================================
    # URL VALIDATION
    # ================================================================
    
    def validate_and_clean_urls(
        self, 
        remove_invalid: bool = True,
        max_check: int = 100,
        timeout: int = 5
    ) -> pd.DataFrame:
        """
        Validate URLs and optionally remove rows with invalid URLs.
        
        Args:
            remove_invalid: Whether to remove rows with invalid URLs
            max_check: Maximum number of unique URLs to check
            timeout: Request timeout in seconds
            
        Returns:
            DataFrame with validated URLs
        """
        logger.info("Validating URLs...")
        
        if 'pageurl' not in self.df.columns:
            logger.warning("No 'pageurl' column found")
            return self.df
            
        # Get unique URLs
        unique_urls = self.df['pageurl'].dropna().unique().tolist()
        
        # Sample URLs if too many
        if len(unique_urls) > max_check:
            logger.info(f"Sampling {max_check} of {len(unique_urls)} URLs for validation")
            import random
            urls_to_check = random.sample(unique_urls, max_check)
        else:
            urls_to_check = unique_urls
            
        logger.info(f"Checking {len(urls_to_check)} unique URLs...")
        
        # Check URLs in parallel
        valid_urls = set()
        invalid_urls = set()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {
                executor.submit(self._check_url, url, timeout): url 
                for url in urls_to_check
            }
            
            for future in tqdm(as_completed(future_to_url), total=len(urls_to_check), desc="Validating URLs"):
                url = future_to_url[future]
                try:
                    is_valid = future.result()
                    if is_valid:
                        valid_urls.add(url)
                    else:
                        invalid_urls.add(url)
                except Exception:
                    invalid_urls.add(url)
                    
        self.invalid_urls = list(invalid_urls)
        
        logger.info(f"URL validation results:")
        logger.info(f"  Valid: {len(valid_urls)}")
        logger.info(f"  Invalid: {len(invalid_urls)}")
        
        # Remove rows with invalid URLs
        if remove_invalid and invalid_urls:
            before_count = len(self.df)
            self.df = self.df[~self.df['pageurl'].isin(invalid_urls)]
            removed = before_count - len(self.df)
            logger.info(f"  Removed {removed} rows with invalid URLs")
            
        return self.df
        
    def _check_url(self, url: str, timeout: int = 5) -> bool:
        """
        Check if a URL is valid and accessible.
        
        Args:
            url: URL to check
            timeout: Request timeout
            
        Returns:
            True if URL is valid
        """
        try:
            # Quick HEAD request to check if URL exists
            response = requests.head(
                url, 
                timeout=timeout, 
                allow_redirects=True,
                headers={'User-Agent': self.config.scraper.user_agent}
            )
            
            # Valid if status code is 2xx or 3xx
            return response.status_code < 400
            
        except requests.RequestException:
            return False
        
    # ================================================================
    # WEB SCRAPING
    # ================================================================
        
    def impute_from_scraping(self, max_urls: Optional[int] = None) -> pd.DataFrame:
        """
        Fill missing values by scraping Walmart product pages.
        
        Args:
            max_urls: Maximum number of URLs to scrape
            
        Returns:
            DataFrame with scraped values filled
        """
        logger.info("Imputing from web scraping...")
        
        # Find URLs with missing reviews
        if 'pageurl' not in self.df.columns or 'review' not in self.df.columns:
            logger.warning("Missing required columns for scraping")
            return self.df
            
        # Get unique URLs where review is missing
        missing_review_mask = self.df['review'].isna()
        urls_to_scrape = self.df.loc[missing_review_mask, 'pageurl'].dropna().unique().tolist()
        
        # Filter out already known invalid URLs
        urls_to_scrape = [url for url in urls_to_scrape if url not in self.invalid_urls]
        
        if max_urls:
            urls_to_scrape = urls_to_scrape[:max_urls]
            
        if not urls_to_scrape:
            logger.info("No URLs to scrape for missing reviews")
            return self.df
            
        logger.info(f"Scraping {len(urls_to_scrape)} unique URLs for missing reviews...")
        
        try:
            with WalmartScraper() as scraper:
                results = scraper.scrape_batch(urls_to_scrape)
                
            # Cache results
            success_count = 0
            for result in results:
                if result.get('success'):
                    self.scraped_cache[result['url']] = result
                    success_count += 1
                else:
                    # Mark as invalid
                    self.invalid_urls.append(result['url'])
                    
            logger.info(f"Successfully scraped {success_count}/{len(urls_to_scrape)} URLs")
            
            # Fill values from scraped data
            self._fill_from_scraped()
            
        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            logger.info("Scraping unavailable, will use LLM-based imputation")
            
        return self.df
        
    def _fill_from_scraped(self) -> None:
        """Fill DataFrame values from scraped cache."""
        filled_counts = {'title': 0, 'rating': 0, 'review_count': 0}
        
        for idx, row in self.df.iterrows():
            url = row.get('pageurl')
            if not url or url not in self.scraped_cache:
                continue
                
            scraped = self.scraped_cache[url]
            
            # Fill title if missing
            if pd.isna(row.get('title')) and 'title' in scraped:
                self.df.at[idx, 'title'] = scraped['title']
                filled_counts['title'] += 1
                
            # Fill rating if missing
            if pd.isna(row.get('rating')) and 'rating' in scraped:
                self.df.at[idx, 'rating'] = scraped['rating']
                filled_counts['rating'] += 1
                
        for col, count in filled_counts.items():
            if count > 0:
                logger.info(f"Filled {count} values in '{col}' from scraping")
                
    # ================================================================
    # LLM INFERENCE
    # ================================================================
                
    def impute_from_llm(self) -> pd.DataFrame:
        """
        Fill missing values using LLM inference.
        
        Uses Gemini to infer missing data from context.
        
        Returns:
            DataFrame with LLM-inferred values
        """
        logger.info("Imputing from LLM (Gemini)...")
        
        # Check if Gemini API is available
        if not self.config.gemini.api_key:
            logger.warning("Gemini API key not set, skipping LLM imputation")
            return self.df
            
        try:
            from src.clustering.gemini_client import GeminiClient
            client = GeminiClient()
            
            if not client.is_available:
                logger.warning("Gemini client not available")
                return self.df
            
            # Find rows with missing title but have review or pageurl
            if 'title' in self.df.columns:
                missing_title_mask = self.df['title'].isna() | (self.df['title'] == 'Unknown Product')
                has_context = (
                    self.df['review'].notna() | 
                    self.df['pageurl'].notna()
                )
                rows_to_infer = self.df[missing_title_mask & has_context]
                
                if len(rows_to_infer) > 0:
                    logger.info(f"Inferring title for {len(rows_to_infer)} rows...")
                    
                    # Batch inference (sample for efficiency)
                    sample_size = min(50, len(rows_to_infer))
                    sample_indices = rows_to_infer.sample(sample_size).index
                    
                    filled_count = 0
                    for idx in tqdm(sample_indices, desc="LLM Inference"):
                        row = self.df.loc[idx]
                        
                        # Build context from URL
                        context_parts = []
                        if pd.notna(row.get('pageurl')):
                            # Extract product name from URL
                            url = str(row['pageurl'])
                            if '/ip/' in url:
                                parts = url.split('/ip/')[-1].split('/')
                                if parts:
                                    product_slug = parts[0].replace('-', ' ')
                                    context_parts.append(f"Product URL slug: {product_slug}")
                                    
                        if pd.notna(row.get('review')):
                            review_snippet = str(row['review'])[:200]
                            context_parts.append(f"Review: {review_snippet}")
                            
                        if context_parts:
                            prompt = (
                                f"Based on this context, what is the product name? "
                                f"Return ONLY the product name, nothing else.\n\n"
                                f"{' | '.join(context_parts)}"
                            )
                            
                            try:
                                title = client.generate(prompt, max_tokens=50)
                                if title and len(title) < 100 and title.lower() != 'unknown':
                                    self.df.at[idx, 'title'] = title.strip()
                                    filled_count += 1
                            except Exception as e:
                                logger.debug(f"LLM inference failed for idx {idx}: {e}")
                                continue
                                
                    logger.info(f"Filled {filled_count} titles via LLM inference")
                        
        except ImportError:
            logger.warning("GeminiClient not available")
        except Exception as e:
            logger.error(f"LLM imputation failed: {e}")
            
        return self.df
        
    # ================================================================
    # DEFAULT FILLING
    # ================================================================
        
    def fill_defaults(self) -> pd.DataFrame:
        """
        Fill remaining missing values with defaults.
        
        Returns:
            DataFrame with default values
        """
        logger.info("Filling remaining missing values with defaults...")
        
        # Fill numeric columns with 0 or median
        numeric_cols = [
            'review_upvotes', 'review_downvotes', 
            'five_star', 'four_star', 'three_star', 'two_star', 'one_star'
        ]
        
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0)
                
        # Fill rating with median if available
        if 'rating' in self.df.columns:
            median_rating = self.df['rating'].median()
            self.df['rating'] = self.df['rating'].fillna(median_rating)
            
        # Fill boolean columns
        if 'verified_purchaser' in self.df.columns:
            self.df['verified_purchaser'] = self.df['verified_purchaser'].fillna('Unknown')
            
        if 'recommended_purchase' in self.df.columns:
            self.df['recommended_purchase'] = self.df['recommended_purchase'].fillna('Unknown')
            
        # Fill text with placeholder
        if 'title' in self.df.columns:
            self.df['title'] = self.df['title'].fillna('Unknown Product')
            
        if 'reviewer_name' in self.df.columns:
            self.df['reviewer_name'] = self.df['reviewer_name'].fillna('Anonymous')
            
        return self.df
        
    # ================================================================
    # REPORTING
    # ================================================================
        
    def get_imputation_report(self) -> Dict[str, Any]:
        """
        Generate report on imputation results.
        
        Returns:
            Dictionary with imputation statistics
        """
        report = {
            'total_rows': len(self.df),
            'scraped_urls': len(self.scraped_cache),
            'scraped_success': sum(1 for v in self.scraped_cache.values() if v.get('success')),
            'invalid_urls': len(self.invalid_urls),
            'columns': {}
        }
        
        for col in self.df.columns:
            null_count = self.df[col].isna().sum()
            report['columns'][col] = {
                'null_count': null_count,
                'null_percent': round(null_count / len(self.df) * 100, 2)
            }
            
        return report
        
    def get_invalid_urls(self) -> List[str]:
        """Get list of invalid URLs found during validation."""
        return self.invalid_urls
