"""
Abstract base scraper class with crawl, parse, and fill pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """
    Abstract base class for web scrapers.

    Scraping pipeline:
    1. crawl   : fetch raw data
    2. parse   : convert raw data into structured fields
    3. fill    : fill missing fields using parsed result
    """

    # ---------- ABSTRACT METHODS (must be implemented) ----------

    @abstractmethod
    def crawl(self, url: str) -> Dict[str, Any]:
        """
        Crawl raw data from a URL (HTML, JSON, text, etc.).
        """
        pass

    @abstractmethod
    def parse(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse raw data into structured product information.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        pass

    # ---------- OPTIONAL OVERRIDES ----------

    def required_fields(self) -> List[str]:
        """
        Fields that must be filled.
        Child classes may override.
        """
        return []

    def allow_overwrite(self) -> bool:
        """
        Whether parsed data can overwrite existing fields.
        """
        return False

    # ---------- SHARED FILLING LOGIC ----------

    def fill(
        self,
        base_record: Dict[str, Any],
        new_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fill missing fields in base_record using new_data.
        """
        result = dict(base_record)

        for key, value in new_data.items():
            if value in (None, "", []):
                continue

            if self.allow_overwrite():
                result[key] = value
            else:
                if result.get(key) in (None, "", []):
                    result[key] = value

        return result

    def is_complete(self, record: Dict[str, Any]) -> bool:
        """
        Check if all required fields are filled.
        """
        for field in self.required_fields():
            if record.get(field) in (None, "", []):
                return False
        return True

    # ---------- TEMPLATE METHOD ----------

    def scrape(self, url: str, record: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Full scraping pipeline (Template Method).

        This method should NOT be overridden.
        """
        logger.info(f"Scraping URL: {url}")

        record = record or {"url": url}

        raw_data = self.crawl(url)
        parsed = self.parse(raw_data)

        filled = self.fill(record, parsed)

        filled["success"] = self.is_complete(filled)
        filled["source"] = self.__class__.__name__

        return filled

    # ---------- CONTEXT MANAGER ----------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
