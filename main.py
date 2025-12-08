"""
Walmart Product Review Analysis Pipeline

Main entry point for the analysis pipeline.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config.settings import settings, DATA_DIR, OUTPUT_DIR
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.imputer import DataImputer
from src.clustering.product_clusterer import ProductClusterer
from src.analysis.aspect_extractor import AspectExtractor
from src.analysis.sentiment_analyzer import SentimentAnalyzer
from src.analysis.insight_generator import InsightGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / 'pipeline.log')
    ]
)

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """
    Main analysis pipeline orchestrator.
    
    Coordinates all pipeline steps:
    1. Load data from Kaggle
    2. Preprocess & clean
    3. Fill missing values
    4. Cluster products
    5. Extract aspects & analyze sentiment
    6. Generate insights & report
    """
    
    def __init__(self):
        """Initialize pipeline."""
        self.df: Optional[pd.DataFrame] = None
        self.loader = DataLoader()
        
    def run(
        self,
        step: Optional[str] = None,
        use_scraping: bool = False,
        max_scrape_urls: int = 50
    ) -> pd.DataFrame:
        """
        Run the analysis pipeline.
        
        Args:
            step: Optional specific step to run (load, preprocess, scrape, cluster, analyze, report)
            use_scraping: Whether to use web scraping for missing data
            max_scrape_urls: Maximum URLs to scrape
            
        Returns:
            Final processed DataFrame
        """
        steps = ['load', 'preprocess', 'scrape', 'cluster', 'analyze', 'report']
        
        if step and step not in steps:
            raise ValueError(f"Invalid step: {step}. Must be one of {steps}")
            
        # Determine which steps to run
        if step:
            step_idx = steps.index(step)
            # Load previous data for steps after 'load'
            if step_idx > 0:
                self._load_saved_data()
            steps_to_run = [step]
        else:
            steps_to_run = steps
            
        logger.info(f"Running pipeline steps: {steps_to_run}")
        
        for s in steps_to_run:
            if s == 'load':
                self._step_load()
            elif s == 'preprocess':
                self._step_preprocess()
            elif s == 'scrape':
                self._step_scrape(use_scraping, max_scrape_urls)
            elif s == 'cluster':
                self._step_cluster()
            elif s == 'analyze':
                self._step_analyze()
            elif s == 'report':
                self._step_report()
                
        return self.df
        
    def _load_saved_data(self) -> None:
        """Load previously saved data."""
        processed_path = settings.processed_data_path
        
        if processed_path.exists():
            logger.info(f"Loading saved data from {processed_path}")
            self.df = pd.read_csv(processed_path)
        else:
            logger.warning("No saved data found, starting from load step")
            self._step_load()
            
    def _step_load(self) -> None:
        """Step 1: Load data from Kaggle."""
        logger.info("=" * 50)
        logger.info("STEP 1: Loading Data")
        logger.info("=" * 50)
        
        self.df = self.loader.load_from_kaggle()
        
        # Save raw data
        self.loader.save_to_csv(self.df, settings.raw_data_path)
        
        logger.info(f"Loaded {len(self.df)} records")
        logger.info(f"Columns: {list(self.df.columns)}")
        
    def _step_preprocess(self) -> None:
        """Step 2: Preprocess data."""
        logger.info("=" * 50)
        logger.info("STEP 2: Preprocessing Data")
        logger.info("=" * 50)
        
        if self.df is None:
            self._load_saved_data()
            
        preprocessor = DataPreprocessor(self.df)
        self.df = preprocessor.preprocess_all()
        
        # Show missing value summary
        missing_info = preprocessor.get_missing_info()
        if len(missing_info) > 0:
            logger.info("Missing values summary:")
            for _, row in missing_info.iterrows():
                logger.info(f"  {row['column']}: {row['missing_count']} ({row['missing_percent']}%)")
                
        # Save processed data
        self.loader.save_to_csv(self.df, settings.processed_data_path)
        
    def _step_scrape(self, use_scraping: bool, max_urls: int, validate_urls: bool = True) -> None:
        """Step 3: Fill missing values."""
        logger.info("=" * 50)
        logger.info("STEP 3: Validating URLs & Filling Missing Values")
        logger.info("=" * 50)
        
        if self.df is None:
            self._load_saved_data()
            
        imputer = DataImputer(self.df)
        self.df = imputer.impute_all(
            use_scraping=use_scraping,
            use_llm=True,
            validate_urls=validate_urls,
            max_scrape_urls=max_urls,
            remove_invalid=True  # Remove rows with invalid URLs
        )
        
        # Report
        report = imputer.get_imputation_report()
        logger.info(f"Imputation complete:")
        logger.info(f"  Scraped: {report['scraped_urls']} URLs")
        logger.info(f"  Invalid URLs removed: {report['invalid_urls']}")
        
        # Save
        self.loader.save_to_csv(self.df, settings.processed_data_path)
        
    def _step_cluster(self) -> None:
        """Step 4: Cluster products."""
        logger.info("=" * 50)
        logger.info("STEP 4: Clustering Products")
        logger.info("=" * 50)
        
        if self.df is None:
            self._load_saved_data()
            
        clusterer = ProductClusterer(self.df)
        self.df = clusterer.cluster_products()
        self.df = clusterer.consolidate_categories(min_count=10)
        
        # Show category summary
        summary = clusterer.get_category_summary()
        logger.info("Category summary:")
        logger.info(f"\n{summary.head(10)}")
        
        # Save
        self.loader.save_to_csv(self.df, settings.clustered_data_path)
        
    def _step_analyze(self) -> None:
        """Step 5: Analyze reviews."""
        logger.info("=" * 50)
        logger.info("STEP 5: Analyzing Reviews")
        logger.info("=" * 50)
        
        if self.df is None:
            self._load_saved_data()
            
        # Extract aspects
        aspect_extractor = AspectExtractor(self.df)
        self.df = aspect_extractor.extract_aspects()
        
        aspect_summary = aspect_extractor.get_aspect_summary()
        logger.info("Aspect summary:")
        logger.info(f"\n{aspect_summary}")
        
        # Analyze sentiment
        sentiment_analyzer = SentimentAnalyzer(self.df)
        self.df = sentiment_analyzer.analyze_overall_sentiment()
        
        category_sentiment = sentiment_analyzer.analyze_by_category()
        if len(category_sentiment) > 0:
            logger.info("Category sentiment:")
            logger.info(f"\n{category_sentiment.head(10)}")
            
        # Save
        self.loader.save_to_csv(self.df, settings.sentiment_data_path)
        
    def _step_report(self) -> None:
        """Step 6: Generate report."""
        logger.info("=" * 50)
        logger.info("STEP 6: Generating Report")
        logger.info("=" * 50)
        
        if self.df is None:
            self._load_saved_data()
            
        insight_generator = InsightGenerator(self.df)
        insights = insight_generator.generate_all_insights()
        
        # Generate and save report
        report = insight_generator.generate_report(settings.report_path)
        
        logger.info(f"Report saved to {settings.report_path}")
        
        # Print recommendations
        logger.info("\nTop Recommendations:")
        for rec in insights.get('recommendations', [])[:3]:
            logger.info(f"  [{rec['priority'].upper()}] {rec['area']}: {rec['recommendation']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Walmart Product Review Analysis Pipeline'
    )
    parser.add_argument(
        '--step',
        choices=['load', 'preprocess', 'scrape', 'cluster', 'analyze', 'report'],
        help='Run specific step only'
    )
    parser.add_argument(
        '--scrape',
        action='store_true',
        help='Enable web scraping for missing data'
    )
    parser.add_argument(
        '--max-scrape-urls',
        type=int,
        default=50,
        help='Maximum URLs to scrape (default: 50)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    logger.info("Starting Walmart Review Analysis Pipeline")
    logger.info(f"Configuration: {args}")
    
    pipeline = AnalysisPipeline()
    
    try:
        df = pipeline.run(
            step=args.step,
            use_scraping=args.scrape,
            max_scrape_urls=args.max_scrape_urls
        )
        
        logger.info("=" * 50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)
        logger.info(f"Final dataset shape: {df.shape}")
        logger.info(f"Output files saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
