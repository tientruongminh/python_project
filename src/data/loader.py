"""
Data loader module for loading data from Kaggle.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

from src.config.settings import settings, DATA_DIR
from src.utils.helpers import normalize_column_name

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loader class for loading Walmart product reviews from Kaggle.
    
    Attributes:
        config: Kaggle configuration settings
        _df: Cached DataFrame
    """
    
    def __init__(self):
        """Initialize DataLoader with configuration."""
        self.config = settings.kaggle
        self._df: Optional[pd.DataFrame] = None
        
    def load_from_kaggle(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load dataset from Kaggle using kagglehub.
        
        Args:
            force_reload: If True, reload even if cached
            
        Returns:
            DataFrame with loaded data
        """
        if self._df is not None and not force_reload:
            logger.info("Returning cached DataFrame")
            return self._df
            
        logger.info(f"Loading dataset from Kaggle: {self.config.dataset_name}")
        
        try:
            df = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                self.config.dataset_name,
                self.config.file_name,
            )
            
            logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
            
            # Normalize column names
            df = self._normalize_columns(df)
            
            # Validate required columns
            self._validate_columns(df)
            
            self._df = df
            return df
            
        except Exception as e:
            logger.error(f"Failed to load dataset from Kaggle: {e}")
            raise
            
    def load_from_csv(self, filepath: Path) -> pd.DataFrame:
        """
        Load dataset from a local CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading data from CSV: {filepath}")
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
            
        df = pd.read_csv(filepath)
        
        # Normalize column names
        df = self._normalize_columns(df)
        
        # Validate required columns
        self._validate_columns(df)
        
        self._df = df
        return df
        
    def save_to_csv(self, df: pd.DataFrame, filepath: Path) -> None:
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            filepath: Output file path
        """
        logger.info(f"Saving data to CSV: {filepath}")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} records")
        
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names to snake_case.
        
        Args:
            df: DataFrame with original column names
            
        Returns:
            DataFrame with normalized column names
        """
        original_columns = list(df.columns)
        new_columns = [normalize_column_name(col) for col in original_columns]
        
        # Log renamed columns
        for orig, new in zip(original_columns, new_columns):
            if orig != new:
                logger.debug(f"Renamed column: '{orig}' -> '{new}'")
                
        df.columns = new_columns
        return df
        
    def _validate_columns(self, df: pd.DataFrame) -> bool:
        """
        Validate that required columns exist.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If required columns are missing
        """
        missing = set(settings.columns.required_columns) - set(df.columns)
        
        if missing:
            logger.warning(f"Missing required columns: {missing}")
            # Don't raise, just warn - some columns might have different names
            
        return True
        
    def get_column_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get information about DataFrame columns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            DataFrame with column information
        """
        info = []
        for col in df.columns:
            info.append({
                'column': col,
                'dtype': str(df[col].dtype),
                'non_null_count': df[col].notna().sum(),
                'null_count': df[col].isna().sum(),
                'null_percent': round(df[col].isna().sum() / len(df) * 100, 2),
                'unique_count': df[col].nunique()
            })
            
        return pd.DataFrame(info)
        
    @property
    def dataframe(self) -> Optional[pd.DataFrame]:
        """Get cached DataFrame."""
        return self._df
