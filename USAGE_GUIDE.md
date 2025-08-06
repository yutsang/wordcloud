# Sentiment Phrase Processor - Usage Guide

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Test the installation**:
   ```bash
   python test_processor.py
   ```

3. **Run a demo**:
   ```bash
   python demo.py
   ```

4. **Configure your API key**:
   - Edit `config.py` and add your API key to `DEEPSEEK_API_KEY = "your_key_here"`
   - Or leave it empty and enter when prompted

5. **Process your data**:
   ```bash
   python sentiment_processor.py
   ```

## What This Program Does

### 1. Translation
- **Detects non-English phrases** using character analysis
- **Translates to English** using DeepSeek API
- **Handles mixed language** content (e.g., "easy to use 非常好用")

### 2. Phrase Merging
- **Identifies similar phrases** using string similarity algorithms
- **Merges phrases** with similarity score ≥ 0.8 (configurable)
- **Adds up frequencies** of merged phrases
- **Keeps shorter phrases** when merging (configurable)

### 3. Word Cloud Generation
- **Creates visual word clouds** for each bank and sentiment
- **Word size** corresponds to frequency
- **High-resolution PNG** output (1200x800, 300 DPI)

## Example Processing

### Input Phrases:
```
'easy to use' (frequency: 10)
'easy to' (frequency: 5)
'very good' (frequency: 8)
'very good app' (frequency: 3)
'非常好用' (frequency: 2)
```

### After Translation:
```
'easy to use' (frequency: 10)
'easy to' (frequency: 5)
'very good' (frequency: 8)
'very good app' (frequency: 3)
'very easy to use' (frequency: 2)  # Translated from Chinese
```

### After Merging:
```
'easy to use' (frequency: 17)  # Merged 'easy to' + 'easy to use'
'very good' (frequency: 11)    # Merged 'very good' + 'very good app'
'very easy to use' (frequency: 2)
```

## Configuration

Edit `config.py` to customize:

```python
# Phrase merging sensitivity
SIMILARITY_THRESHOLD = 0.8  # Higher = more strict merging

# Word cloud appearance
WORDCLOUD_MAX_WORDS = 100
WORDCLOUD_COLORMAP = 'viridis'

# API settings
API_TIMEOUT = 30
API_RETRY_DELAY = 1
```

## File Structure

```
sentiment-noGH/
├── sentiment_processor.py    # Main program
├── config.py                 # Configuration settings
├── requirements.txt          # Dependencies
├── test_processor.py         # Test suite
├── demo.py                   # Demo script
├── README.md                 # Documentation
├── USAGE_GUIDE.md           # This file
├── mox_bank_positive_phrases.txt
├── mox_bank_negative_phrases.txt
├── welab_bank_positive_phrases.txt
├── welab_bank_negative_phrases.txt
├── za_bank_positive_phrases.txt
├── za_bank_negative_phrases.txt
├── processed_phrases.json    # Output: processed results
└── wordclouds/              # Output: generated word clouds
    ├── mox_bank_positive_wordcloud.png
    ├── mox_bank_negative_wordcloud.png
    ├── welab_bank_positive_wordcloud.png
    ├── welab_bank_negative_wordcloud.png
    ├── za_bank_positive_wordcloud.png
    └── za_bank_negative_wordcloud.png
```

## API Key Setup

1. **Sign up** at [DeepSeek Platform](https://platform.deepseek.com/)
2. **Generate API key** from your dashboard
3. **Configure your API key** (choose one option):
   - **Option A**: Add your API key to `config.py`:
     ```python
     DEEPSEEK_API_KEY = "your_api_key_here"
     ```
   - **Option B**: Enter it when prompted by the program

## Troubleshooting

### Common Issues:

1. **Import Errors**:
   ```bash
   pip install -r requirements.txt
   ```

2. **API Key Issues**:
   - Ensure key is valid and has credits
   - Check internet connection
   - Verify API endpoint is accessible

3. **Missing Files**:
   - Ensure all 6 phrase files are present
   - Check file naming convention

4. **Memory Issues**:
   - Reduce `BATCH_SIZE` in config.py
   - Process files individually

### Error Messages:

- `"Translation failed"`: API error, original phrase kept
- `"File not found"`: Missing phrase file
- `"No phrases to generate wordcloud"`: Empty or invalid data

## Advanced Usage

### Custom Similarity Threshold:
```python
from sentiment_processor import SentimentProcessor

processor = SentimentProcessor("your_api_key")
results = processor.process_all_files(similarity_threshold=0.9)  # More strict
```

### Process Individual Files:
```python
# Parse single file
phrases = processor.parse_phrase_file("my_phrases.txt")

# Translate phrases
translated = {}
for phrase, freq in phrases.items():
    translated[processor.translate_to_english(phrase)] = freq

# Merge similar phrases
merged = processor.merge_similar_phrases(translated, 0.8)

# Generate word cloud
processor.generate_wordcloud(merged, "My Phrases", "my_wordcloud.png")
```

### Custom Word Cloud:
```python
# Generate custom word cloud
wordcloud = WordCloud(
    width=800,
    height=600,
    background_color='black',
    colormap='plasma',
    max_words=50
).generate_from_frequencies(phrases)
```

## Performance Tips

1. **Batch Processing**: Process files in smaller batches for large datasets
2. **API Optimization**: Adjust `API_RETRY_DELAY` based on rate limits
3. **Memory Management**: Use `SAVE_INTERMEDIATE_RESULTS = True` for large files
4. **Similarity Threshold**: Higher values reduce processing time but may miss merges

## Output Formats

### JSON Output (`processed_phrases.json`):
```json
{
  "mox_bank": {
    "positive": {
      "easy to use": 105,
      "user friendly": 90,
      "very good": 77
    },
    "negative": {
      "not working": 24,
      "failed to": 18,
      "app crashes": 16
    }
  }
}
```

### Word Cloud Images:
- **Format**: PNG
- **Resolution**: 1200x800 pixels
- **DPI**: 300
- **Color Scheme**: Viridis (configurable)

## Support

For issues or questions:
1. Run `python test_processor.py` to check installation
2. Check the logs for detailed error messages
3. Verify your API key and internet connection
4. Ensure all required files are present

## License

This project is open source and available under the MIT License. 