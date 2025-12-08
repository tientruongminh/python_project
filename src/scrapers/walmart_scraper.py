"""
Walmart web scraper using Selenium.
"""
from __future__ import annotations

import logging
import re
from time import sleep
from typing import Any, Dict, List, Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    WebDriverException
)
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

from src.scrapers.base_scraper import BaseScraper
from src.config.settings import settings
from src.utils.helpers import retry_with_backoff, clean_text

logger = logging.getLogger(__name__)


class WalmartScraper(BaseScraper):
    """
    Selenium-based scraper for Walmart product pages.
    
    Handles:
    - Headless Chrome browser automation
    - Rate limiting to avoid blocks
    - Retry logic for failed requests
    - Product info extraction
    """
    
    def __init__(self, config=None):
        """
        Initialize Walmart scraper.
        
        Args:
            config: Optional ScraperConfig override
        """
        self.config = config or settings.scraper
        self.driver: Optional[webdriver.Chrome] = None
        self._setup_driver()
        
    def _setup_driver(self) -> None:
        """Set up Chrome WebDriver with options."""
        logger.info("Setting up Chrome WebDriver...")
        
        options = Options()
        
        if self.config.headless:
            options.add_argument("--headless=new")
            
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument(f"--user-agent={self.config.user_agent}")
        options.add_argument("--disable-blink-features=AutomationControlled")
        
        # Suppress logging
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
            self.driver.implicitly_wait(10)
            logger.info("Chrome WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise
            
    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def scrape(self, url: str) -> Dict[str, Any]:
        """
        Scrape product data from Walmart URL.
        
        Args:
            url: Walmart product page URL
            
        Returns:
            Dictionary with scraped product data
        """
        if not self.driver:
            raise RuntimeError("WebDriver not initialized")
            
        logger.info(f"Scraping: {url}")
        
        try:
            # Navigate to page
            self.driver.get(url)
            
            # Wait for page to load
            WebDriverWait(self.driver, self.config.timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Rate limiting
            sleep(self.config.rate_limit_delay)
            
            # Extract data from page
            page_source = self.driver.page_source
            data = self.extract_product_info(page_source)
            data['url'] = url
            data['success'] = True
            
            return data
            
        except TimeoutException:
            logger.warning(f"Timeout waiting for page: {url}")
            return {'url': url, 'success': False, 'error': 'timeout'}
            
        except WebDriverException as e:
            logger.error(f"WebDriver error for {url}: {e}")
            return {'url': url, 'success': False, 'error': str(e)}
            
    def extract_product_info(self, page_source: str) -> Dict[str, Any]:
        """
        Extract product information from Walmart page HTML.
        
        Args:
            page_source: HTML source of the page
            
        Returns:
            Dictionary with:
            - title: Product title
            - rating: Average rating
            - review_count: Number of reviews
            - price: Product price
            - description: Product description
        """
        soup = BeautifulSoup(page_source, 'html.parser')
        data = {}
        
        # Extract title
        title_selectors = [
            'h1[itemprop="name"]',
            'h1.prod-ProductTitle',
            'h1[data-automation-id="product-title"]',
            'h1.lh-copy'
        ]
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                data['title'] = clean_text(element.get_text())
                break
                
        # Extract rating
        rating_selectors = [
            'span[itemprop="ratingValue"]',
            'span.rating-number',
            'span[data-automation-id="product-rating"]'
        ]
        
        for selector in rating_selectors:
            element = soup.select_one(selector)
            if element:
                rating_text = element.get_text()
                rating_match = re.search(r'(\d+\.?\d*)', rating_text)
                if rating_match:
                    data['rating'] = float(rating_match.group(1))
                break
                
        # Extract review count
        review_count_selectors = [
            'span[itemprop="reviewCount"]',
            'a[data-automation-id="product-reviews"]',
            'span.review-count'
        ]
        
        for selector in review_count_selectors:
            element = soup.select_one(selector)
            if element:
                count_text = element.get_text()
                count_match = re.search(r'(\d+)', count_text.replace(',', ''))
                if count_match:
                    data['review_count'] = int(count_match.group(1))
                break
                
        # Extract price
        price_selectors = [
            'span[itemprop="price"]',
            'span.price-characteristic',
            'span[data-automation-id="product-price"]'
        ]
        
        for selector in price_selectors:
            element = soup.select_one(selector)
            if element:
                price_text = element.get_text()
                price_match = re.search(r'(\d+\.?\d*)', price_text.replace(',', ''))
                if price_match:
                    data['price'] = float(price_match.group(1))
                break
                
        # Extract description
        desc_selectors = [
            'div[itemprop="description"]',
            'div.about-product',
            'div[data-automation-id="product-description"]'
        ]
        
        for selector in desc_selectors:
            element = soup.select_one(selector)
            if element:
                data['description'] = clean_text(element.get_text())[:500]  # Limit length
                break
                
        return data
        
    def scrape_batch(self, urls: List[str], max_urls: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Scrape multiple URLs.
        
        Args:
            urls: List of URLs to scrape
            max_urls: Maximum number of URLs to scrape
            
        Returns:
            List of scraped data dictionaries
        """
        if max_urls:
            urls = urls[:max_urls]
            
        logger.info(f"Starting batch scrape of {len(urls)} URLs")
        
        results = []
        success_count = 0
        
        for i, url in enumerate(urls):
            try:
                result = self.scrape(url)
                results.append(result)
                
                if result.get('success'):
                    success_count += 1
                    
                # Progress logging every 10 URLs
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i + 1}/{len(urls)} ({success_count} successful)")
                    
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {e}")
                results.append({'url': url, 'success': False, 'error': str(e)})
                
        logger.info(f"Batch scrape complete: {success_count}/{len(urls)} successful")
        return results
        
    def close(self) -> None:
        """Close the WebDriver."""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("WebDriver closed")
            except Exception as e:
                logger.warning(f"Error closing WebDriver: {e}")
            finally:
                self.driver = None
