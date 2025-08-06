# Sentiment Phrase Processor

This program processes sentiment phrases from multiple banks, translates non-English phrases to English using DeepSeek API, merges similar phrases, and generates word clouds.

## Features

- **Translation**: Automatically detects and translates non-English phrases to English using DeepSeek API
- **Phrase Merging**: Merges similar phrases and adds up their frequencies
- **Word Cloud Generation**: Creates beautiful word clouds for each bank's positive and negative phrases
- **Multi-bank Support**: Processes phrases from Mox Bank, WeLab Bank, and ZA Bank
- **Comprehensive Logging**: Detailed logging of all processing steps

## Requirements

- Python 3.7+
- DeepSeek API key
- Internet connection for API calls

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. **Get a DeepSeek API Key**:
   - Sign up at [DeepSeek](https://platform.deepseek.com/)
   - Generate an API key from your dashboard

2. **Configure your API key** (choose one option):
   - **Option A**: Add your API key to `config.py`:
     ```python
     DEEPSEEK_API_KEY = "your_api_key_here"
     ```
   - **Option B**: Enter it when prompted by the program

3. **Prepare your phrase files**:
   - Ensure you have the following files in the same directory:
     - `mox_bank_positive_phrases.txt`
     - `mox_bank_negative_phrases.txt`
     - `welab_bank_positive_phrases.txt`
     - `welab_bank_negative_phrases.txt`
     - `za_bank_positive_phrases.txt`
     - `za_bank_negative_phrases.txt`

4. **Run the program**:

```bash
python sentiment_processor.py
```

## Output

The program will generate:

1. **Processed Results**: `processed_phrases.json` - Contains all processed phrases with translations and merged frequencies
2. **Word Clouds**: A `wordclouds/` directory containing PNG images for each bank and sentiment:
   - `mox_bank_positive_wordcloud.png`
   - `mox_bank_negative_wordcloud.png`
   - `welab_bank_positive_wordcloud.png`
   - `welab_bank_negative_wordcloud.png`
   - `za_bank_positive_wordcloud.png`
   - `za_bank_negative_wordcloud.png`

## How It Works

### 1. Translation Process
- Detects non-English text using character analysis
- Uses DeepSeek API to translate phrases to English
- Handles API errors gracefully by keeping original text

### 2. Phrase Merging
- Uses string similarity algorithms to identify similar phrases
- Merges phrases with similarity score ≥ 0.8 (configurable)
- Adds up frequencies of merged phrases
- Keeps the shorter phrase when merging (or higher frequency if equal)

### 3. Word Cloud Generation
- Creates visually appealing word clouds
- Size of words corresponds to frequency
- Uses viridis color scheme
- Saves high-resolution PNG images

## Configuration

You can modify the following parameters in the code:

- `similarity_threshold`: Threshold for merging similar phrases (default: 0.8)
- Word cloud settings: size, colors, max words, etc.
- API timeout and retry settings

## Example Output

```
============================================================
PROCESSING SUMMARY
============================================================

Mox Bank:
  Positive: 245 unique phrases, 1855 total frequency
    Top phrases:
      'easy to' (frequency: 105)
      'user friendly' (frequency: 90)
      'very good' (frequency: 77)
      'very convenient' (frequency: 38)
      'great app' (frequency: 33)

  Negative: 156 unique phrases, 12516 total frequency
    Top phrases:
      'not working' (frequency: 24)
      'failed to' (frequency: 18)
      'account opening' (frequency: 18)
      'app crashes' (frequency: 16)
      'customer service' (frequency: 14)
```

## Error Handling

- **API Errors**: If translation fails, the original phrase is kept
- **File Errors**: Missing files are logged and skipped
- **Network Issues**: Timeout handling and retry logic
- **Rate Limiting**: Built-in delays between API calls

## Logging

The program provides detailed logging including:
- Translation progress
- Phrase merging statistics
- File processing status
- Error messages and warnings

## Troubleshooting

1. **API Key Issues**: Ensure your DeepSeek API key is valid and has sufficient credits
2. **Missing Files**: Check that all phrase files are in the correct location
3. **Network Issues**: Ensure stable internet connection for API calls
4. **Memory Issues**: For large files, consider processing in batches

## License

This project is open source and available under the MIT License. 