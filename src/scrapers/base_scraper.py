"""
Abstract base scraper class.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """
    Abstract base class for web scrapers.
    
    Define the interface for all scraper implementations.
    """
    
    @abstractmethod
    def scrape(self, url: str) -> Dict[str, Any]:
        """
        Scrape data from a URL.
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary with scraped data
        """
        pass
        
    @abstractmethod
    def extract_product_info(self, page_source: str) -> Dict[str, Any]:
        """
        Extract product information from page HTML.
        
        Args:
            page_source: HTML source of the page
            
        Returns:
            Dictionary with extracted data
        """
        pass
        
    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        pass
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
