#!/usr/bin/env python3
"""
Welab Bank Negative - Final Cleanup
- Remove/translate Chinese characters
- Adjust font sizes to match attached image (more visible words)
- Use correct color RGB(0, 184, 245)
- Ensure better word distribution
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

class WelabNegativeFinalCleanup:
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

    def translate_chinese_characters(self, phrases):
        def translate_phrase(phrase):
            # Check if phrase contains Chinese characters or square boxes
            if re.search(r'[\u4e00-\u9fff]|□', phrase):
                prompt = f"Translate this text to English, keeping only meaningful content: '{phrase}'"
                try:
                    payload = {
                        "model": "deepseek-chat",
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 100,
                        "temperature": 0.1
                    }
                    response = self.session.post(self.api_base_url, headers=self.headers, json=payload, timeout=30)
                    response.raise_for_status()
                    result = response.json()
                    translated = result['choices'][0]['message']['content'].strip().strip('"\'')
                    return translated if translated and len(translated) >= 3 else None
                except Exception:
                    return None
            return phrase

        print("Translating Chinese characters...")
        translated_phrases = {}
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(translate_phrase, phrase): phrase for phrase in phrases.keys()}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Translating"):
                original_phrase = futures[future]
                try:
                    translated = future.result()
                    if translated:
                        translated_phrases[translated] = translated_phrases.get(translated, 0) + phrases[original_phrase]
                except Exception:
                    translated_phrases[original_phrase] = phrases[original_phrase]
        
        return translated_phrases

    def clean_and_merge(self, phrases):
        # Remove very short or problematic phrases
        cleaned_phrases = {}
        for phrase, freq in phrases.items():
            # Clean the phrase - remove newlines, extra spaces, and problematic characters
            clean_phrase = re.sub(r'\s+', ' ', phrase.strip())
            clean_phrase = re.sub(r'[^\w\s\-.,!?]', '', clean_phrase)  # Keep only alphanumeric, spaces, and basic punctuation
            
            if len(clean_phrase) >= 3 and not re.search(r'^[□\s]+$', clean_phrase):
                cleaned_phrases[clean_phrase] = cleaned_phrases.get(clean_phrase, 0) + freq
        
        # Merge very similar phrases
        def merge_similar(phrases_dict):
            prompt = f"""Merge these similar negative phrases about Welab Bank, combining frequencies. Keep meaningful phrases and remove duplicates:\n\n"""
            phrase_list = [f"'{p}' (frequency: {f})" for p, f in list(phrases_dict.items())[:50]]
            prompt += "\n".join(phrase_list)
            prompt += "\n\nReturn as JSON array: [{'phrase': 'merged phrase', 'frequency': total_frequency}, ...]"
            
            try:
                payload = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 800,
                    "temperature": 0.2
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

        print("Cleaning and merging similar phrases...")
        return merge_similar(cleaned_phrases)

    def save_phrases(self, phrases, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("Welab Bank Negative - Final Cleanup Phrases\n")
            f.write("="*50 + "\n\n")
            for phrase, freq in sorted(phrases.items(), key=lambda x: x[1], reverse=True):
                f.write(f"'{phrase}' (frequency: {freq})\n")

    def generate_wordcloud(self, phrases, filename):
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            return f"rgb({self.negative_color[0]}, {self.negative_color[1]}, {self.negative_color[2]})"
        
        # Adjust parameters to match attached image - more visible words, better distribution
        max_words = min(200, len(phrases))  # More words visible
        min_font_size = 12  # Smaller minimum for more words
        max_font_size = 120  # Reasonable maximum
        relative_scaling = 0.7  # Better distribution
        
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
            regexp=r'\b\w+\b'  # Only word boundaries
        ).generate_from_frequencies(phrases)
        
        plt.figure(figsize=(14, 14))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    api_key = DEEPSEEK_API_KEY
    processor = WelabNegativeFinalCleanup(api_key)
    
    # Use the balanced content file as input
    input_file = 'processed_phrases/welab_bank_negative_balanced_content_processed_phrases.txt'
    output_file = 'processed_phrases/welab_bank_negative_final_cleanup_phrases.txt'
    wordcloud_file = 'wordclouds/welab_bank_negative_final_cleanup_wordcloud.png'
    
    phrases = processor.parse_phrases(input_file)
    print(f"Original phrases: {len(phrases)}")
    
    # Step 1: Translate Chinese characters
    translated_phrases = processor.translate_chinese_characters(phrases)
    print(f"After translation: {len(translated_phrases)}")
    
    # Step 2: Clean and merge similar phrases
    final_phrases = processor.clean_and_merge(translated_phrases)
    print(f"After cleanup: {len(final_phrases)}")
    
    processor.save_phrases(final_phrases, output_file)
    processor.generate_wordcloud(final_phrases, wordcloud_file)
    
    print(f"Final cleanup complete!")
    print(f"Output: {output_file}")
    print(f"Wordcloud: {wordcloud_file}") 