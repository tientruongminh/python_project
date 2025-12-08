"""
Unit tests for data preprocessor.
"""
import pytest
import pandas as pd
from datetime import datetime

from src.data.preprocessor import DataPreprocessor
from src.utils.helpers import to_snake_case, normalize_column_name


class TestToSnakeCase:
    """Tests for to_snake_case function."""
    
    def test_camel_case(self):
        assert to_snake_case("camelCase") == "camel_case"
        
    def test_pascal_case(self):
        assert to_snake_case("PascalCase") == "pascal_case"
        
    def test_with_spaces(self):
        assert to_snake_case("with spaces") == "with_spaces"
        
    def test_mixed(self):
        assert to_snake_case("ReviewUpvotes") == "review_upvotes"
        
    def test_special_chars(self):
        assert to_snake_case("five_star") == "five_star"
        

class TestNormalizeColumnName:
    """Tests for normalize_column_name function."""
    
    def test_known_mapping(self):
        assert normalize_column_name("uniq _id") == "uniq_id"
        
    def test_review_upvotes(self):
        assert normalize_column_name("review _upvotes") == "review_upvotes"
        
    def test_normal_column(self):
        assert normalize_column_name("rating") == "rating"


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'uniq_id': ['1', '2', '3'],
            'title': ['Product A', 'Product B', None],
            'rating': [5.0, 4.0, 3.0],
            'review': ['Great product!', 'Good quality', 'Okay'],
            'review_date': ['2020-04-15', '2020-05-20', '2020-06-10'],
            'pageurl': [
                'https://walmart.com/ip/A/123',
                'https://walmart.com/ip/B/456',
                'https://walmart.com/ip/A/123'  # Duplicate URL
            ],
            'review_upvotes': [10, 5, None],
            'review_downvotes': [2, 1, 0]
        })
        
    def test_convert_data_types(self, sample_df):
        preprocessor = DataPreprocessor(sample_df)
        result = preprocessor.convert_data_types()
        
        assert result['rating'].dtype == 'float64'
        assert pd.api.types.is_datetime64_any_dtype(result['review_date'])
        
    def test_add_derived_columns(self, sample_df):
        preprocessor = DataPreprocessor(sample_df)
        preprocessor.convert_data_types()
        result = preprocessor.add_derived_columns()
        
        assert 'product_id' in result.columns
        assert 'total_votes' in result.columns
        assert 'review_length' in result.columns
        
    def test_merge_duplicate_urls(self, sample_df):
        # Add None to title for duplicate URL
        sample_df.loc[2, 'title'] = None
        
        preprocessor = DataPreprocessor(sample_df)
        result = preprocessor.merge_duplicate_urls()
        
        # Title should be filled from duplicate URL
        assert result.loc[2, 'title'] == 'Product A'
        
    def test_get_missing_info(self, sample_df):
        preprocessor = DataPreprocessor(sample_df)
        missing_info = preprocessor.get_missing_info()
        
        assert len(missing_info) > 0
        assert 'column' in missing_info.columns
        assert 'missing_count' in missing_info.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
