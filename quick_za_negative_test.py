#!/usr/bin/env python3
"""
Quick Test - ZA Bank Negative Only
Test the processing for ZA Bank negative to verify everything works
"""
import os
import re
import json
import requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import *

class QuickZANegativeTest:
    def __init__(self, api_key, num_workers=15):
        self.api_key = api_key
        self.api_base_url = DEEPSEEK_API_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.num_workers = num_workers
        self.negative_color = (0, 184, 245)
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=num_workers, pool_maxsize=num_workers)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def parse_phrases(self, filepath):
        phrases = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                m = re.match(r"'(.+)' \(frequency: (\d+)\)", line.strip())
                if m:
                    phrase, freq = m.groups()
                    phrases[phrase] = int(freq)
        return phrases

    def targeted_cleanup(self, phrases):
        """Targeted cleanup for negative phrases"""
        problematic_patterns = [
            r'^[□\s]+$',  # Only square boxes or spaces
            r'^[a-z]$',   # Single letters
            r'^[0-9]+$',  # Only numbers
            r'^[^\w\s]+$', # Only special characters
        ]
        
        # Phrases to remove for negative sentiment
        remove_phrases = {
            'good experience', 'works well', 'very good', 'excellent service',
            'user friendly', 'easy to', 'convenient', 'great app',
            'perfect', 'awesome', 'nice', 'smooth', 'helpful'
        }
        
        cleaned_phrases = {}
        for phrase, freq in phrases.items():
            # Clean the phrase
            clean_phrase = re.sub(r'\s+', ' ', phrase.strip())
            
            # Skip if matches problematic patterns
            skip = False
            for pattern in problematic_patterns:
                if re.match(pattern, clean_phrase):
                    skip = True
                    break
            
            if skip:
                continue
                
            # Skip if in remove list
            if clean_phrase.lower() in remove_phrases:
                continue
                
            # Keep the phrase
            if len(clean_phrase) >= 2:
                cleaned_phrases[clean_phrase] = cleaned_phrases.get(clean_phrase, 0) + freq
        
        return cleaned_phrases

    def merge_obvious_duplicates(self, phrases):
        """Merge only very obvious duplicates"""
        def merge_similar(phrases_dict):
            prompt = f"""Merge ONLY very obvious duplicates from these negative phrases about ZA Bank. Only merge phrases that are essentially the same meaning. Keep most phrases as they are.\n\n"""
            phrase_list = [f"'{p}' (frequency: {f})" for p, f in list(phrases_dict.items())[:50]]
            prompt += "\n".join(phrase_list)
            prompt += "\n\nReturn as JSON array: [{'phrase': 'merged phrase', 'frequency': total_frequency}, ...]"
            prompt += "\nOnly merge if they are clearly the same meaning. Keep most phrases unchanged."
            
            try:
                payload = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 800,
                    "temperature": 0.1
                }
                response = self.session.post(self.api_base_url, headers=self.headers, json=payload, timeout=45)
                response.raise_for_status()
                result = response.json()
                ai_response = result['choices'][0]['message']['content'].strip()
                try:
                    merged = json.loads(ai_response)
                    return {item['phrase']: item['frequency'] for item in merged if 'phrase' in item and 'frequency' in item}
                except Exception:
                    return phrases_dict
            except Exception:
                return phrases_dict

        print("Merging obvious duplicates...")
        return merge_similar(phrases)

    def save_phrases(self, phrases, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"ZA Bank Negative - Quick Test Phrases\n")
            f.write("="*50 + "\n\n")
            for phrase, freq in sorted(phrases.items(), key=lambda x: x[1], reverse=True):
                f.write(f"'{phrase}' (frequency: {freq})\n")

    def generate_wordcloud(self, phrases, filename):
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            return f"rgb({self.negative_color[0]}, {self.negative_color[1]}, {self.negative_color[2]})"
        
        # Use 100 max words as requested
        max_words = min(100, len(phrases))
        min_font_size = 14
        max_font_size = 140
        relative_scaling = 0.8
        
        wordcloud = WordCloud(
            width=1200,
            height=1200,
            background_color='white',
            max_words=max_words,
            relative_scaling=relative_scaling,
            collocations=False,
            prefer_horizontal=0.7,
            min_font_size=min_font_size,
            max_font_size=max_font_size,
            color_func=color_func,
            regexp=r'\b\w+\b'
        ).generate_from_frequencies(phrases)
        
        plt.figure(figsize=(14, 14))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    api_key = DEEPSEEK_API_KEY
    if not api_key:
        print("❌ Please set your DeepSeek API key in config.py")
        exit(1)
    
    processor = QuickZANegativeTest(api_key)
    
    # Input and output files
    input_file = 'processed_phrases/za_bank_negative_balanced_content_processed_phrases.txt'
    output_file = 'processed_phrases/za_bank_negative_quick_test_phrases.txt'
    wordcloud_file = 'wordclouds/za_bank_negative_quick_test_wordcloud.png'
    
    print("🚀 Quick Test - ZA Bank Negative")
    print("="*40)
    
    phrases = processor.parse_phrases(input_file)
    print(f"Original phrases: {len(phrases)}")
    
    # Step 1: Targeted cleanup
    cleaned_phrases = processor.targeted_cleanup(phrases)
    print(f"After cleanup: {len(cleaned_phrases)}")
    
    # Step 2: Merge obvious duplicates
    final_phrases = processor.merge_obvious_duplicates(cleaned_phrases)
    print(f"After merging: {len(final_phrases)}")
    
    # Save and generate wordcloud
    processor.save_phrases(final_phrases, output_file)
    processor.generate_wordcloud(final_phrases, wordcloud_file)
    
    print(f"\n✅ Quick test complete!")
    print(f"📁 Phrases: {output_file}")
    print(f"🖼️  Wordcloud: {wordcloud_file}")
    print(f"📊 Max words: 100")
    print(f"🎨 Color: RGB({processor.negative_color[0]}, {processor.negative_color[1]}, {processor.negative_color[2]})") 