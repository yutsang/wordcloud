# API Key Setup Guide

## Quick Setup

### Option 1: Add to Config File (Recommended)

1. **Open `config.py`** in your text editor
2. **Find this line**:
   ```python
   DEEPSEEK_API_KEY = ""  # Example: "sk-1234567890abcdef..."
   ```
3. **Replace the empty string** with your API key:
   ```python
   DEEPSEEK_API_KEY = "sk-your-actual-api-key-here"
   ```
4. **Save the file**
5. **Run the program**:
   ```bash
   python sentiment_processor.py
   ```

### Option 2: Enter When Prompted

1. **Leave `config.py` unchanged** (keep `DEEPSEEK_API_KEY = ""`)
2. **Run the program**:
   ```bash
   python sentiment_processor.py
   ```
3. **Enter your API key** when prompted

## Getting Your DeepSeek API Key

1. **Sign up** at [DeepSeek Platform](https://platform.deepseek.com/)
2. **Log in** to your account
3. **Go to API Keys** section
4. **Create a new API key**
5. **Copy the key** (it starts with "sk-")

## Example Config File

```python
# config.py
DEEPSEEK_API_KEY = "sk-1234567890abcdef1234567890abcdef"  # Your actual key here
DEEPSEEK_API_BASE_URL = "https://api.deepseek.com/v1/chat/completions"
API_TIMEOUT = 30
API_RETRY_DELAY = 1
# ... rest of config
```

## Security Note

- **Never commit your API key** to version control
- **Keep your API key private** and don't share it
- **Consider using environment variables** for production use

## Troubleshooting

- **"API key is required"**: Make sure you've added the key to config.py or entered it when prompted
- **"Invalid API key"**: Check that your key is correct and has sufficient credits
- **"API rate limit exceeded"**: Wait a moment and try again, or check your API usage 