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

# Optional import for Selenium scraper
try:
    from src.scrapers.walmart_scraper import WalmartScraper
    SELENIUM_AVAILABLE = True
except ImportError:
    WalmartScraper = None
    SELENIUM_AVAILABLE = False

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
        
        Uses Gemini to infer missing data from context (reviews).
        Target: 
        - Title: 'Unknown Product' or NaN
        - Rating: 0 or NaN
        
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
            
            # Identify target rows
            # 1. Missing Title
            mask_title = pd.Series([False] * len(self.df), index=self.df.index)
            if 'title' in self.df.columns:
                mask_title = self.df['title'].isna() | (self.df['title'] == 'Unknown Product')
            
            # 2. Missing/Zero Rating
            mask_rating = pd.Series([False] * len(self.df), index=self.df.index)
            if 'rating' in self.df.columns:
                mask_rating = self.df['rating'].isna() | (self.df['rating'] == 0)
                
            # Combine: Rows that need EITHER title OR rating fix, AND have review text
            mask_needs_infer = (mask_title | mask_rating)
            if 'review' in self.df.columns:
                 mask_needs_infer &= self.df['review'].notna() & (self.df['review'].str.len() > 10)
            else:
                mask_needs_infer = pd.Series([False] * len(self.df))

            rows_to_infer = self.df[mask_needs_infer]
            
            if len(rows_to_infer) == 0:
                logger.info("No rows require LLM inference")
                return self.df
                
            logger.info(f"Inferring data for {len(rows_to_infer)} rows (Title/Rating)...")
            
            # Prepare batches
            indices = rows_to_infer.index.tolist()
            reviews = rows_to_infer['review'].tolist()
            
            # Infer in batches using new method
            inferred_results = client.infer_product_info_batch(reviews)
            
            # Update DataFrame
            filled_title = 0
            filled_rating = 0
            
            for idx, result in zip(indices, inferred_results):
                # Update title if needed
                current_title = self.df.at[idx, 'title'] if 'title' in self.df.columns else None
                if result.get('title') and (pd.isna(current_title) or current_title == 'Unknown Product'):
                     if 'title' in self.df.columns:
                        self.df.at[idx, 'title'] = result['title']
                        filled_title += 1
                
                # Update rating if needed
                current_rating = self.df.at[idx, 'rating'] if 'rating' in self.df.columns else 0
                if result.get('rating') and (pd.isna(current_rating) or current_rating == 0):
                    if 'rating' in self.df.columns:
                        self.df.at[idx, 'rating'] = float(result['rating'])
                        filled_rating += 1
                        
            logger.info(f"LLM Inference Results: Filled {filled_title} titles, {filled_rating} ratings")
                        
        except ImportError:
            logger.warning("GeminiClient not available")
        except Exception as e:
            logger.error(f"LLM imputation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
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
    # GEMINI URL CONTEXT SCRAPING
    # ================================================================
    
    def impute_with_gemini_url(
        self,
        max_urls: Optional[int] = None,
        columns_to_fill: List[str] = None,
        progress_callback = None
    ) -> pd.DataFrame:
        """
        Fill missing values bằng Gemini URL Context API.
        
        Đây là phương pháp nhanh và ổn định hơn Selenium.
        
        Args:
            max_urls: Maximum URLs to process
            columns_to_fill: Columns to fill (default: title, star distributions)
            progress_callback: Optional callback(current, total, url)
            
        Returns:
            DataFrame with filled values
        """
        logger.info("Imputing with Gemini URL Context API...")
        
        if columns_to_fill is None:
            columns_to_fill = ['title', 'five_star', 'four_star', 'three_star', 
                              'two_star', 'one_star']
        
        # Initialize scraper
        try:
            from src.scrapers.gemini_url_scraper import GeminiURLScraper
            scraper = GeminiURLScraper()
            
            if not scraper.is_available:
                logger.error("Gemini URL Scraper không khả dụng")
                return self.df
                
        except ImportError as e:
            logger.error(f"Không thể import GeminiURLScraper: {e}")
            return self.df
        
        # Find URLs with missing data
        url_col = 'pageurl' if 'pageurl' in self.df.columns else 'url'
        if url_col not in self.df.columns:
            logger.warning("Không tìm thấy cột URL")
            return self.df
        
        # Get unique URLs where title is missing
        if 'title' in columns_to_fill and 'title' in self.df.columns:
            missing_mask = self.df['title'].isna()
        else:
            # Or where any star column is missing
            star_cols = [c for c in columns_to_fill if 'star' in c and c in self.df.columns]
            if star_cols:
                missing_mask = self.df[star_cols[0]].isna()
            else:
                logger.info("Không có columns cần fill")
                return self.df
        
        urls_to_process = self.df.loc[missing_mask, url_col].dropna().unique().tolist()
        
        # Filter out invalid URLs
        urls_to_process = [u for u in urls_to_process if u not in self.invalid_urls]
        
        if max_urls:
            urls_to_process = urls_to_process[:max_urls]
        
        if not urls_to_process:
            logger.info("Không có URLs cần process")
            return self.df
        
        logger.info(f"Processing {len(urls_to_process)} URLs với Gemini...")
        
        # Process URLs
        results = {}
        total = len(urls_to_process)
        
        for i, url in enumerate(urls_to_process):
            if progress_callback:
                progress_callback(i + 1, total, url)
            
            result = scraper.scrape_product(url)
            
            if result.error:
                logger.debug(f"Error scraping {url}: {result.error}")
                self.invalid_urls.append(url)
            else:
                results[url] = result
            
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i + 1}/{total} ({len(results)} successful)")
        
        logger.info(f"Scraped {len(results)}/{total} URLs successfully")
        
        # Apply results to DataFrame
        filled_count = 0
        
        for url, product_info in results.items():
            mask = self.df[url_col] == url
            
            if 'title' in columns_to_fill and product_info.title:
                self.df.loc[mask & self.df['title'].isna(), 'title'] = product_info.title
                filled_count += mask.sum()
            
            # Star distributions
            if 'five_star' in columns_to_fill and product_info.five_star_pct is not None:
                self.df.loc[mask & self.df['five_star'].isna(), 'five_star'] = product_info.five_star_pct
            if 'four_star' in columns_to_fill and product_info.four_star_pct is not None:
                self.df.loc[mask & self.df['four_star'].isna(), 'four_star'] = product_info.four_star_pct
            if 'three_star' in columns_to_fill and product_info.three_star_pct is not None:
                self.df.loc[mask & self.df['three_star'].isna(), 'three_star'] = product_info.three_star_pct
            if 'two_star' in columns_to_fill and product_info.two_star_pct is not None:
                self.df.loc[mask & self.df['two_star'].isna(), 'two_star'] = product_info.two_star_pct
            if 'one_star' in columns_to_fill and product_info.one_star_pct is not None:
                self.df.loc[mask & self.df['one_star'].isna(), 'one_star'] = product_info.one_star_pct
        
        logger.info(f"Filled {filled_count} rows with scraped data")
        
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

