"""
Configuration settings for the sentiment processor
"""

# API Configuration
# Add your DeepSeek API key here, or leave empty to enter it when prompted
DEEPSEEK_API_KEY = "sk-013e00392a7d433b8d1d09d88bc0b62e"  # Example: "sk-1234567890abcdef..."
DEEPSEEK_API_BASE_URL = "https://api.deepseek.com/v1/chat/completions"
API_TIMEOUT = 30  # seconds
API_RETRY_DELAY = 1  # seconds between API calls

# Translation Configuration
ENGLISH_DETECTION_THRESHOLD = 0.8  # Minimum ASCII ratio to consider text as English
TRANSLATION_MAX_TOKENS = 100
TRANSLATION_TEMPERATURE = 0.1

# Phrase Merging Configuration
SIMILARITY_THRESHOLD = 0.8  # Minimum similarity score to merge phrases
MERGE_PREFER_SHORTER = True  # Prefer shorter phrases when merging

# Word Cloud Configuration
WORDCLOUD_WIDTH = 1200
WORDCLOUD_HEIGHT = 800
WORDCLOUD_MAX_WORDS = 100
WORDCLOUD_COLORMAP = 'viridis'
WORDCLOUD_RELATIVE_SCALING = 0.5
WORDCLOUD_DPI = 300

# File Configuration
OUTPUT_DIR = "wordclouds"
PROCESSED_RESULTS_FILE = "processed_phrases.json"

# Banks and Sentiments
BANKS = ['mox_bank', 'welab_bank', 'za_bank']
SENTIMENTS = ['positive', 'negative']

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Processing Configuration
BATCH_SIZE = 50  # Number of phrases to process in a batch
SAVE_INTERMEDIATE_RESULTS = True  # Save results after each bank is processed 