# Walmart Product Review Analysis Pipeline

A comprehensive data analysis pipeline for Walmart product reviews with OOP architecture, web scraping, Gemini AI clustering, and aspect-based sentiment analysis.

##  Features

- **Data Loading**: Automatic loading from Kaggle with column normalization
- **Preprocessing**:  duplicate URL merging, missing value handling
- **Web Scraping**: Selenium-based scraper for Walmart product pages
- **AI Clustering**: Product categorization using Google Gemini API
- **Sentiment Analysis**: Aspect-based sentiment analysis of reviews
- **Business Insights**: Actionable recommendations based on analysis

##  Project Structure

```
python_project/
├── .streamlit/
│   └── config.toml                          # Streamlit configuration
│
├── data/
│   ├── raw_data.csv                         # Raw Walmart reviews dataset
│   ├── processed_data.csv                   # Cleaned dataset (version 1)
│   └── processed_data_v2.csv                # Cleaned dataset (version 2)
│
├── outputs/
│   ├── analysis_report.md                   # Generated business insights report
│   ├── clustered_products.csv               # Products with assigned categories
│   ├── sentiment_analysis.csv               # Aspect-based sentiment results
│   └── pipeline.log                         # Pipeline execution logs
│
├── src/
│   ├── __init__.py                          # Package initializer
│   │
│   ├── analysis/
│   │   ├── __init__.py                      # Analysis module init
│   │   ├── aspect_extractor.py              # Extract aspects from reviews
│   │   ├── aspect_summarizer.py             # Summarize aspects per category
│   │   ├── evaluator.py                     # Model evaluation metrics
│   │   ├── insight_generator.py             # Generate business insights
│   │   ├── rag_pipeline.py                  # RAG-based Q&A pipeline
│   │   ├── sentiment_analyzer.py            # Sentiment classification
│   │   └── topic_modeler.py                 # Topic modeling with LDA/BERTopic
│   │
│   ├── clustering/
│   │   ├── __init__.py                      # Clustering module init
│   │   ├── gemini_client.py                 # Google Gemini API client
│   │   └── product_clusterer.py             # Product categorization logic
│   │
│   ├── config/
│   │   ├── __init__.py                      # Config module init
│   │   └── settings.py                      # Environment & pipeline settings
│   │
│   ├── data/
│   │   ├── __init__.py                      # Data module init
│   │   ├── loader.py                        # Kaggle data loader
│   │   ├── preprocessor.py                  # Data cleaning & transformation
│   │   └── imputer.py                       # Missing value imputation
│   │
│   ├── scrapers/
│   │   ├── __init__.py                      # Scrapers module init
│   │   ├── base_scraper.py                  # Abstract base scraper class
│   │   └── walmart_scraper.py               # Walmart product page scraper
│   │
│   └── utils/
│       ├── __init__.py                      # Utils module init
│       └── helpers.py                       # Common helper functions
│
├── tests/
│   ├── __init__.py                          # Tests module init
│   └── test_preprocessor.py                 # Preprocessor tests
│
├── .env                                     # Environment variables (API keys)
├── .gitignore                               # Git ignore rules
├── main.py                                  # Pipeline entry point (CLI)
├── streamlit_app.py                         # Interactive dashboard
└── requirements.txt                         # Python dependencies
```

##  Installation

```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

Set your Gemini API key:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
GEMINI_API_KEY=your-api-key-here
```

## 🔧 Usage

### Full Pipeline
```bash
python main.py
```

### Step-by-step
```bash
python main.py --step load          # Load data only
python main.py --step preprocess    # Preprocess data
python main.py --step scrape        # Fill missing via scraping
python main.py --step cluster       # Cluster products
python main.py --step analyze       # Analyze reviews
python main.py --step report        # Generate report
```

##  Output

- `outputs/processed_data.csv` - Cleaned dataset
- `outputs/clustered_products.csv` - Products with categories
- `outputs/sentiment_analysis.csv` - Aspect-based sentiments
- `outputs/analysis_report.md` - Business insights report

## Methodology

1. **Data Preprocessing**
   - Rename columns to snake_case
   - Shift dates back 10 years
   - Merge duplicate PageURLs

2. **Missing Data Imputation**
   - Scrape Walmart pages via Selenium
   - Fallback: LLM inference for missing fields

3. **Product Clustering**
   - Extract product info from PageURLs
   - Use Gemini API to categorize products
   - Assign meaningful category names

4. **Aspect-Based Sentiment Analysis**
   - Extract aspects: quality, price, shipping, etc.
   - Analyze sentiment per aspect per category
   - Track sentiment trends over time

5. **Insight Generation**
   - Identify customer pain points
   - Generate actionable recommendations
   - Create business strategy suggestions

## 📄 License

MIT License
