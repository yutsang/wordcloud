#!/usr/bin/env python3
"""
General Sentiment Processor
- Processes all 3 banks (Mox, Welab, ZA) for both positive and negative sentiments
- Uses relative paths for easy execution
- Always updates the same output files
- Max 100 words per word cloud
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

class GeneralSentimentProcessor:
    def __init__(self, api_key, num_workers=15):
        self.api_key = api_key
        self.api_base_url = DEEPSEEK_API_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.num_workers = num_workers
        self.positive_color = (12, 35, 60)
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

    def targeted_cleanup(self, phrases, sentiment):
        """Targeted cleanup based on sentiment"""
        problematic_patterns = [
            r'^[□\s]+$',  # Only square boxes or spaces
            r'^[a-z]$',   # Single letters
            r'^[0-9]+$',  # Only numbers
            r'^[^\w\s]+$', # Only special characters
        ]
        
        # Phrases to remove based on sentiment
        if sentiment == 'positive':
            remove_phrases = {
                'bad experience', 'terrible', 'awful', 'horrible', 'worst',
                'cannot login', 'unable to login', 'login problems', 'error',
                'slow', 'crashes', 'broken', 'unreliable'
            }
        else:  # negative
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

    def merge_obvious_duplicates(self, phrases, sentiment):
        """Merge only very obvious duplicates"""
        def merge_similar(phrases_dict):
            prompt = f"""Merge ONLY very obvious duplicates from these {sentiment} phrases. Only merge phrases that are essentially the same meaning. Keep most phrases as they are.\n\n"""
            phrase_list = [f"'{p}' (frequency: {f})" for p, f in list(phrases_dict.items())[:100]]
            prompt += "\n".join(phrase_list)
            prompt += "\n\nReturn as JSON array: [{'phrase': 'merged phrase', 'frequency': total_frequency}, ...]"
            prompt += "\nOnly merge if they are clearly the same meaning. Keep most phrases unchanged."
            
            try:
                payload = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1000,
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

        print(f"Merging obvious duplicates for {sentiment}...")
        return merge_similar(phrases)

    def save_phrases(self, phrases, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Processed Phrases\n")
            f.write("="*50 + "\n\n")
            for phrase, freq in sorted(phrases.items(), key=lambda x: x[1], reverse=True):
                f.write(f"'{phrase}' (frequency: {freq})\n")

    def generate_wordcloud(self, phrases, sentiment, bank, filename):
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            if sentiment == 'positive':
                return f"rgb({self.positive_color[0]}, {self.positive_color[1]}, {self.positive_color[2]})"
            else:
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

    def process_bank(self, bank, sentiment):
        """Process a single bank and sentiment"""
        print(f"\n=== Processing {bank} Bank {sentiment} ===")
        
        # Input file (use the most recent processed file)
        input_file = f'processed_phrases/{bank}_bank_{sentiment}_balanced_content_processed_phrases.txt'
        
        # Output files (always the same)
        output_file = f'processed_phrases/{bank}_bank_{sentiment}_final_phrases.txt'
        wordcloud_file = f'wordclouds/{bank}_bank_{sentiment}_wordcloud.png'
        
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Input file not found: {input_file}")
            return None
        
        phrases = self.parse_phrases(input_file)
        print(f"Original phrases: {len(phrases)}")
        
        # Step 1: Targeted cleanup
        cleaned_phrases = self.targeted_cleanup(phrases, sentiment)
        print(f"After cleanup: {len(cleaned_phrases)}")
        
        # Step 2: Merge obvious duplicates
        final_phrases = self.merge_obvious_duplicates(cleaned_phrases, sentiment)
        print(f"After merging: {len(final_phrases)}")
        
        # Save and generate wordcloud
        self.save_phrases(final_phrases, output_file)
        self.generate_wordcloud(final_phrases, sentiment, bank, wordcloud_file)
        
        print(f"✅ {bank} {sentiment} complete!")
        print(f"   Phrases: {output_file}")
        print(f"   Wordcloud: {wordcloud_file}")
        
        return final_phrases

    def process_all_banks(self):
        """Process all banks and sentiments"""
        banks = ['mox', 'welab', 'za']
        sentiments = ['positive', 'negative']
        
        print("🚀 Starting General Sentiment Processing")
        print("="*50)
        
        results = {}
        for bank in banks:
            results[bank] = {}
            for sentiment in sentiments:
                try:
                    phrases = self.process_bank(bank, sentiment)
                    results[bank][sentiment] = phrases
                except Exception as e:
                    print(f"❌ Error processing {bank} {sentiment}: {e}")
                    results[bank][sentiment] = None
        
        # Print summary
        print("\n" + "="*50)
        print("📊 PROCESSING SUMMARY")
        print("="*50)
        for bank in banks:
            for sentiment in sentiments:
                if results[bank][sentiment]:
                    count = len(results[bank][sentiment])
                    print(f"{bank.capitalize()} {sentiment}: {count} phrases")
                else:
                    print(f"{bank.capitalize()} {sentiment}: ❌ Failed")
        
        print("\n✅ All processing complete!")
        print("📁 Check the 'processed_phrases/' and 'wordclouds/' directories for results.")

if __name__ == "__main__":
    api_key = DEEPSEEK_API_KEY
    if not api_key:
        print("❌ Please set your DeepSeek API key in config.py")
        exit(1)
    
    processor = GeneralSentimentProcessor(api_key)
    processor.process_all_banks() 