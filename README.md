# Walmart Product Review Analysis Pipeline

A comprehensive data analysis pipeline for Walmart product reviews with OOP architecture, web scraping, Gemini AI clustering, and aspect-based sentiment analysis.

##  Features

- **Data Loading**: Automatic loading from Kaggle with column normalization
- **Preprocessing**: Date shifting, duplicate URL merging, missing value handling
- **Web Scraping**: Selenium-based scraper for Walmart product pages
- **AI Clustering**: Product categorization using Google Gemini API
- **Sentiment Analysis**: Aspect-based sentiment analysis of reviews
- **Business Insights**: Actionable recommendations based on analysis

##  Project Structure

```
python_project/
├── src/
│   ├── config/          # Configuration settings
│   ├── data/            # Data loading & preprocessing
│   ├── scrapers/        # Web scraping modules
│   ├── clustering/      # Gemini AI clustering
│   ├── analysis/        # Sentiment & insight analysis
│   └── utils/           # Helper utilities
├── data/                # Raw & processed data
├── outputs/             # Analysis results
├── notebooks/           # Jupyter notebooks
├── tests/               # Unit tests
├── main.py              # Entry point
└── requirements.txt
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
