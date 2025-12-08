"""
Data preprocessor module for cleaning and transforming data.

Implements 5 Data Quality Dimensions:
1. Completeness - Điền đầy đủ missing values
2. Accuracy - Sửa các giá trị sai
3. Validity & Consistency - Đảm bảo format đúng
4. Timeliness - Xử lý dates
5. Uniqueness - Xóa trùng lặp
"""
from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

from src.config.settings import settings
from src.utils.helpers import clean_text, extract_product_id_from_url

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Data preprocessor for cleaning and transforming Walmart review data.
    
    Implements complete EDA pipeline with 5 quality dimensions.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize preprocessor with DataFrame.
        
        Args:
            df: Input DataFrame to process
        """
        self.df = df.copy()
        self.config = settings
        self.quality_report: Dict[str, Any] = {}
        
    def preprocess_all(self) -> pd.DataFrame:
        """
        Run all preprocessing steps following 5 Data Quality Dimensions.
        
        Returns:
            Fully preprocessed DataFrame
        """
        logger.info("=" * 60)
        logger.info("STARTING COMPREHENSIVE DATA PREPROCESSING")
        logger.info("=" * 60)
        
        initial_shape = self.df.shape
        
        # ============================================================
        # DIMENSION 1: COMPLETENESS - Điền đầy đủ dữ liệu
        # ============================================================
        logger.info("\n[1/5] COMPLETENESS - Xử lý missing values...")
        self.clean_text_columns()
        self.convert_data_types()
        self.merge_duplicate_urls()
        self.fill_missing_from_aggregation()
        
        # ============================================================
        # DIMENSION 2: ACCURACY - Sửa giá trị sai
        # ============================================================
        logger.info("\n[2/5] ACCURACY - Kiểm tra và sửa giá trị không hợp lệ...")
        self.fix_invalid_values()
        
        # ============================================================
        # DIMENSION 3: VALIDITY & CONSISTENCY - Đảm bảo format đúng
        # ============================================================
        logger.info("\n[3/5] VALIDITY - Chuẩn hóa format...")
        self.standardize_categorical_columns()
        self.validate_rating_range()
        
        # ============================================================
        # DIMENSION 4: TIMELINESS - Xử lý thời gian
        # ============================================================
        logger.info("\n[4/5] TIMELINESS - Xử lý dates...")
        self.shift_dates()
        self.validate_dates()
        
        # ============================================================
        # DIMENSION 5: UNIQUENESS - Xóa trùng lặp
        # ============================================================
        logger.info("\n[5/5] UNIQUENESS - Xóa dữ liệu trùng lặp...")
        self.remove_duplicates()
        
        # ============================================================
        # FINAL: Add derived columns
        # ============================================================
        logger.info("\n[FINAL] Adding derived columns...")
        self.add_derived_columns()
        
        # Generate quality report
        self._generate_quality_report(initial_shape)
        
        logger.info("=" * 60)
        logger.info(f"PREPROCESSING COMPLETE: {initial_shape} -> {self.df.shape}")
        logger.info("=" * 60)
        
        return self.df

    # ================================================================
    # DIMENSION 1: COMPLETENESS
    # ================================================================
    
    def clean_text_columns(self) -> pd.DataFrame:
        """Clean text columns by removing extra whitespace and special characters."""
        logger.info("  Cleaning text columns...")
        
        text_columns = ['title', 'review', 'reviewer_name']
        
        for col in text_columns:
            if col in self.df.columns:
                # Clean text
                self.df[col] = self.df[col].apply(
                    lambda x: clean_text(str(x)) if pd.notna(x) else x
                )
                
                # Replace empty strings with NaN
                self.df[col] = self.df[col].replace('', np.nan)
                self.df[col] = self.df[col].replace('nan', np.nan)
                self.df[col] = self.df[col].replace('None', np.nan)
                
        return self.df
        
    def convert_data_types(self) -> pd.DataFrame:
        """Convert columns to appropriate data types."""
        logger.info("  Converting data types...")
        
        # Numeric columns
        numeric_cols = [
            'rating', 'review_upvotes', 'review_downvotes',
            'five_star', 'four_star', 'three_star', 'two_star', 'one_star'
        ]
        
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                
        # Date columns
        date_cols = ['crawl_timestamp', 'review_date']
        
        for col in date_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                
        return self.df
        
    def merge_duplicate_urls(self) -> pd.DataFrame:
        """Fill missing values by using data from other reviews of same product."""
        logger.info("  Merging data from duplicate PageURLs...")
        
        if 'pageurl' not in self.df.columns:
            return self.df
            
        # Columns that can be filled from other reviews of same product
        fillable_cols = ['title', 'five_star', 'four_star', 'three_star', 'two_star', 'one_star']
        
        for col in fillable_cols:
            if col not in self.df.columns:
                continue
                
            before_null = self.df[col].isna().sum()
            
            # Get first non-null value for each URL
            url_values = self.df.groupby('pageurl')[col].first()
            
            # Fill missing values
            missing_mask = self.df[col].isna()
            self.df.loc[missing_mask, col] = self.df.loc[missing_mask, 'pageurl'].map(url_values)
            
            after_null = self.df[col].isna().sum()
            filled = before_null - after_null
            if filled > 0:
                logger.info(f"    Filled {filled} missing values in '{col}'")
                
        return self.df
        
    def fill_missing_from_aggregation(self) -> pd.DataFrame:
        """Fill remaining missing values using statistical methods."""
        logger.info("  Filling remaining missing values...")
        
        # Rating: fill with median
        if 'rating' in self.df.columns:
            median_rating = self.df['rating'].median()
            null_count = self.df['rating'].isna().sum()
            if null_count > 0:
                self.df['rating'] = self.df['rating'].fillna(median_rating)
                logger.info(f"    Filled {null_count} missing ratings with median ({median_rating:.2f})")
                
        # Numeric columns: fill with 0
        numeric_fill_cols = ['review_upvotes', 'review_downvotes', 
                            'five_star', 'four_star', 'three_star', 'two_star', 'one_star']
        for col in numeric_fill_cols:
            if col in self.df.columns:
                null_count = self.df[col].isna().sum()
                if null_count > 0:
                    self.df[col] = self.df[col].fillna(0)
                    logger.info(f"    Filled {null_count} missing '{col}' with 0")
                    
        # Text columns: fill with placeholder
        if 'title' in self.df.columns:
            null_count = self.df['title'].isna().sum()
            if null_count > 0:
                self.df['title'] = self.df['title'].fillna('Unknown Product')
                logger.info(f"    Filled {null_count} missing titles")
                
        if 'reviewer_name' in self.df.columns:
            null_count = self.df['reviewer_name'].isna().sum()
            if null_count > 0:
                self.df['reviewer_name'] = self.df['reviewer_name'].fillna('Anonymous')
                logger.info(f"    Filled {null_count} missing reviewer names")
                
        # Boolean columns
        if 'verified_purchaser' in self.df.columns:
            self.df['verified_purchaser'] = self.df['verified_purchaser'].fillna('Unknown')
            
        if 'recommended_purchase' in self.df.columns:
            self.df['recommended_purchase'] = self.df['recommended_purchase'].fillna('Unknown')
            
        return self.df

    # ================================================================
    # DIMENSION 2: ACCURACY
    # ================================================================
    
    def fix_invalid_values(self) -> pd.DataFrame:
        """Fix invalid values that don't make sense."""
        logger.info("  Fixing invalid values...")
        
        # Fix negative votes
        for col in ['review_upvotes', 'review_downvotes']:
            if col in self.df.columns:
                invalid_count = (self.df[col] < 0).sum()
                if invalid_count > 0:
                    self.df.loc[self.df[col] < 0, col] = 0
                    logger.info(f"    Fixed {invalid_count} negative values in '{col}'")
                    
        # Fix star counts (should be non-negative integers)
        star_cols = ['five_star', 'four_star', 'three_star', 'two_star', 'one_star']
        for col in star_cols:
            if col in self.df.columns:
                # Fix negatives
                self.df.loc[self.df[col] < 0, col] = 0
                # Round to integers
                self.df[col] = self.df[col].round(0).astype(int)
                
        return self.df

    # ================================================================
    # DIMENSION 3: VALIDITY & CONSISTENCY
    # ================================================================
    
    def standardize_categorical_columns(self) -> pd.DataFrame:
        """Standardize categorical columns to consistent format."""
        logger.info("  Standardizing categorical columns...")
        
        # Verified purchaser: standardize to Yes/No/Unknown
        if 'verified_purchaser' in self.df.columns:
            mapping = {
                'true': 'Yes', 'True': 'Yes', 'TRUE': 'Yes', 'yes': 'Yes', 'Yes': 'Yes', '1': 'Yes',
                'false': 'No', 'False': 'No', 'FALSE': 'No', 'no': 'No', 'No': 'No', '0': 'No',
            }
            self.df['verified_purchaser'] = self.df['verified_purchaser'].replace(mapping)
            self.df['verified_purchaser'] = self.df['verified_purchaser'].fillna('Unknown')
            
        # Recommended purchase: standardize
        if 'recommended_purchase' in self.df.columns:
            mapping = {
                'true': 'Yes', 'True': 'Yes', 'yes': 'Yes', 'Yes': 'Yes',
                'false': 'No', 'False': 'No', 'no': 'No', 'No': 'No',
            }
            self.df['recommended_purchase'] = self.df['recommended_purchase'].replace(mapping)
            self.df['recommended_purchase'] = self.df['recommended_purchase'].fillna('Unknown')
            
        # Website column: lowercase and strip
        if 'website' in self.df.columns:
            self.df['website'] = self.df['website'].str.lower().str.strip()
            
        return self.df
        
    def validate_rating_range(self) -> pd.DataFrame:
        """Ensure rating is within valid range [1, 5]."""
        logger.info("  Validating rating range [1-5]...")
        
        if 'rating' in self.df.columns:
            # Clip to valid range
            original_invalid = ((self.df['rating'] < 1) | (self.df['rating'] > 5)).sum()
            
            if original_invalid > 0:
                self.df['rating'] = self.df['rating'].clip(lower=1, upper=5)
                logger.info(f"    Clipped {original_invalid} ratings to [1, 5] range")
                
        return self.df

    # ================================================================
    # DIMENSION 4: TIMELINESS
    # ================================================================
    
    def shift_dates(self, years: Optional[int] = None) -> pd.DataFrame:
        """Shift all date columns back by specified number of years."""
        years = years or self.config.date.years_to_shift
        logger.info(f"  Shifting dates back by {years} years...")
        
        date_cols = self.config.columns.date_columns
        
        for col in date_cols:
            if col in self.df.columns:
                try:
                    # Ensure datetime type
                    if not pd.api.types.is_datetime64_any_dtype(self.df[col]):
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                        
                    # Shift dates
                    valid_dates = self.df[col].notna()
                    self.df.loc[valid_dates, col] = self.df.loc[valid_dates, col].apply(
                        lambda x: x - relativedelta(years=years)
                    )
                    logger.info(f"    Shifted '{col}': {valid_dates.sum()} dates")
                except Exception as e:
                    logger.warning(f"    Failed to shift '{col}': {e}")
                    
        return self.df
        
    def validate_dates(self) -> pd.DataFrame:
        """Validate dates are within reasonable range."""
        logger.info("  Validating date ranges...")
        
        # Define reasonable date range (after shifting)
        min_date = pd.Timestamp('1990-01-01')
        max_date = pd.Timestamp('2015-12-31')  # After 10-year shift from 2020
        
        date_cols = ['crawl_timestamp', 'review_date']
        
        for col in date_cols:
            if col in self.df.columns and pd.api.types.is_datetime64_any_dtype(self.df[col]):
                try:
                    # Convert to tz-naive if tz-aware
                    if self.df[col].dt.tz is not None:
                        self.df[col] = self.df[col].dt.tz_localize(None)
                        
                    # Find invalid dates
                    invalid_mask = (self.df[col] < min_date) | (self.df[col] > max_date)
                    invalid_count = invalid_mask.sum()
                    
                    if invalid_count > 0:
                        # Set invalid dates to NaN
                        self.df.loc[invalid_mask, col] = pd.NaT
                        logger.info(f"    Invalidated {invalid_count} out-of-range dates in '{col}'")
                except Exception as e:
                    logger.warning(f"    Could not validate '{col}': {e}")
                    
        return self.df

    # ================================================================
    # DIMENSION 5: UNIQUENESS
    # ================================================================
    
    def remove_duplicates(self) -> pd.DataFrame:
        """Remove duplicate records based on multiple strategies."""
        logger.info("  Removing duplicate records...")
        
        initial_count = len(self.df)
        
        # Strategy 1: Exact duplicates (all columns)
        self.df = self.df.drop_duplicates()
        after_exact = len(self.df)
        exact_removed = initial_count - after_exact
        if exact_removed > 0:
            logger.info(f"    Removed {exact_removed} exact duplicate rows")
            
        # Strategy 2: Same review from same reviewer on same product
        subset_cols = []
        if 'pageurl' in self.df.columns:
            subset_cols.append('pageurl')
        if 'reviewer_name' in self.df.columns:
            subset_cols.append('reviewer_name')
        if 'review' in self.df.columns:
            subset_cols.append('review')
            
        if len(subset_cols) >= 2:
            before = len(self.df)
            self.df = self.df.drop_duplicates(subset=subset_cols, keep='first')
            after = len(self.df)
            if before - after > 0:
                logger.info(f"    Removed {before - after} duplicate reviews (same user, product, review)")
                
        # Strategy 3: Same uniq_id (should be unique)
        if 'uniq_id' in self.df.columns:
            before = len(self.df)
            self.df = self.df.drop_duplicates(subset=['uniq_id'], keep='first')
            after = len(self.df)
            if before - after > 0:
                logger.info(f"    Removed {before - after} duplicate uniq_ids")
                
        final_count = len(self.df)
        total_removed = initial_count - final_count
        logger.info(f"    Total duplicates removed: {total_removed} ({total_removed/initial_count*100:.2f}%)")
        
        return self.df

    # ================================================================
    # DERIVED COLUMNS
    # ================================================================
    
    def add_derived_columns(self) -> pd.DataFrame:
        """Add derived columns for analysis."""
        logger.info("  Adding derived columns...")
        
        # Extract product ID from URL
        if 'pageurl' in self.df.columns:
            self.df['product_id'] = self.df['pageurl'].apply(extract_product_id_from_url)
            unique_products = self.df['product_id'].nunique()
            logger.info(f"    Extracted {unique_products} unique product IDs")
            
        # Calculate vote metrics
        if 'review_upvotes' in self.df.columns and 'review_downvotes' in self.df.columns:
            self.df['total_votes'] = self.df['review_upvotes'] + self.df['review_downvotes']
            
            self.df['vote_ratio'] = self.df.apply(
                lambda row: row['review_upvotes'] / row['total_votes'] 
                if row['total_votes'] > 0 else 0.5,
                axis=1
            )
            
            # Helpfulness score (Wilson score lower bound)
            self.df['helpfulness_score'] = self.df.apply(
                lambda row: self._wilson_score(row['review_upvotes'], row['total_votes']),
                axis=1
            )
            
        # Calculate review length
        if 'review' in self.df.columns:
            self.df['review_length'] = self.df['review'].apply(
                lambda x: len(str(x)) if pd.notna(x) else 0
            )
            
            # Word count
            self.df['word_count'] = self.df['review'].apply(
                lambda x: len(str(x).split()) if pd.notna(x) else 0
            )
            
        # Categorize rating
        if 'rating' in self.df.columns:
            self.df['rating_category'] = pd.cut(
                self.df['rating'],
                bins=[0, 2, 3.5, 5],
                labels=['Low', 'Medium', 'High'],
                include_lowest=True
            )
            
            # Sentiment label based on rating
            self.df['rating_sentiment'] = pd.cut(
                self.df['rating'],
                bins=[0, 2.5, 3.5, 5],
                labels=['Negative', 'Neutral', 'Positive'],
                include_lowest=True
            )
            
        # Extract year/month for time analysis
        if 'review_date' in self.df.columns:
            self.df['review_year'] = self.df['review_date'].dt.year
            self.df['review_month'] = self.df['review_date'].dt.month
            self.df['review_year_month'] = self.df['review_date'].dt.to_period('M')
            
        return self.df
        
    def _wilson_score(self, upvotes: int, total: int, confidence: float = 0.95) -> float:
        """Calculate Wilson score lower bound for helpfulness ranking."""
        if total == 0:
            return 0.0
            
        z = 1.96  # 95% confidence
        p = upvotes / total
        
        denominator = 1 + z**2 / total
        centre_adjusted_probability = p + z**2 / (2 * total)
        adjusted_standard_deviation = math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
        
        lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
        
        return round(max(0, lower_bound), 4)

    # ================================================================
    # REPORTING & UTILITIES
    # ================================================================
    
    def _generate_quality_report(self, initial_shape: Tuple[int, int]) -> None:
        """Generate data quality report."""
        self.quality_report = {
            'initial_rows': initial_shape[0],
            'final_rows': len(self.df),
            'rows_removed': initial_shape[0] - len(self.df),
            'initial_columns': initial_shape[1],
            'final_columns': len(self.df.columns),
            'completeness': {
                col: round((1 - self.df[col].isna().sum() / len(self.df)) * 100, 2)
                for col in self.df.columns
            },
            'unique_products': self.df['product_id'].nunique() if 'product_id' in self.df.columns else None,
            'date_range': {
                'min': str(self.df['review_date'].min()) if 'review_date' in self.df.columns else None,
                'max': str(self.df['review_date'].max()) if 'review_date' in self.df.columns else None,
            }
        }
        
    def get_quality_report(self) -> Dict[str, Any]:
        """Get the data quality report."""
        return self.quality_report
        
    def get_missing_info(self) -> pd.DataFrame:
        """Get information about missing values."""
        missing_info = []
        
        for col in self.df.columns:
            null_count = self.df[col].isna().sum()
            missing_info.append({
                'column': col,
                'missing_count': null_count,
                'missing_percent': round(null_count / len(self.df) * 100, 2),
                'completeness': round((1 - null_count / len(self.df)) * 100, 2),
                'dtype': str(self.df[col].dtype)
            })
                
        return pd.DataFrame(missing_info).sort_values('missing_count', ascending=False)
        
    def get_urls_with_missing_data(self) -> List[str]:
        """Get URLs that have missing data in scrapable columns."""
        if 'pageurl' not in self.df.columns:
            return []
            
        scrapable_cols = self.config.columns.scrapable_columns
        existing_cols = [col for col in scrapable_cols if col in self.df.columns]
        
        if not existing_cols:
            return []
            
        missing_mask = self.df[existing_cols].isna().any(axis=1)
        urls = self.df.loc[missing_mask, 'pageurl'].dropna().unique().tolist()
        
        logger.info(f"Found {len(urls)} unique URLs with missing scrapable data")
        return urls
        
    def print_summary(self) -> None:
        """Print preprocessing summary."""
        print("\n" + "=" * 60)
        print("DATA QUALITY SUMMARY")
        print("=" * 60)
        
        if self.quality_report:
            print(f"\nRows: {self.quality_report['initial_rows']} → {self.quality_report['final_rows']}")
            print(f"Removed: {self.quality_report['rows_removed']} duplicates")
            print(f"Columns: {self.quality_report['initial_columns']} → {self.quality_report['final_columns']}")
            
            if self.quality_report.get('unique_products'):
                print(f"Unique Products: {self.quality_report['unique_products']}")
                
            print(f"\nDate Range: {self.quality_report['date_range']['min']} to {self.quality_report['date_range']['max']}")
            
            print("\nColumn Completeness (%):")
            for col, pct in sorted(self.quality_report['completeness'].items(), key=lambda x: x[1]):
                status = "✓" if pct >= 95 else "⚠" if pct >= 80 else "✗"
                print(f"  {status} {col}: {pct}%")
                
        print("=" * 60)
