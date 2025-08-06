#!/usr/bin/env python3
"""
Welab Bank Negative - Aggressive Merge
- Aggressively merge similar/related phrases for Welab Bank negative
- Reduce total unique phrases (target: ~180-220)
- Regenerate the word cloud for better visibility
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

class WelabNegativeAggressiveMerge:
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

    def ai_aggressive_merge(self, phrases, target_count=200):
        # Sort by frequency
        sorted_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
        batch_size = 20
        merged_phrases = {}
        def process_batch(batch):
            phrase_list = [f"'{p}' (frequency: {f})" for p, f in batch]
            phrases_text = "\n".join(phrase_list)
            prompt = f"""Aggressively merge these negative sentiment phrases about Welab Bank. Merge similar/related phrases, combine frequencies, and reduce the total number to about {target_count}. Only keep the most meaningful, clear, and representative phrases.\n\nPhrases:\n{phrases_text}\n\nReturn as a JSON array of objects with 'phrase' and 'frequency'.\nExample: [{{'phrase': 'merged phrase', 'frequency': total_frequency}}, ...]\nResult:"""
            try:
                payload = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.2
                }
                response = self.session.post(self.api_base_url, headers=self.headers, json=payload, timeout=45)
                response.raise_for_status()
                result = response.json()
                ai_response = result['choices'][0]['message']['content'].strip()
                try:
                    merged_batch = json.loads(ai_response)
                    for item in merged_batch:
                        if 'phrase' in item and 'frequency' in item:
                            phrase = item['phrase'].strip()
                            if phrase and len(phrase) >= 3:
                                merged_phrases[phrase] = merged_phrases.get(phrase, 0) + item['frequency']
                except Exception:
                    for p, f in batch:
                        merged_phrases[p] = merged_phrases.get(p, 0) + f
            except Exception as e:
                for p, f in batch:
                    merged_phrases[p] = merged_phrases.get(p, 0) + f
        print(f"Aggressively merging {len(sorted_phrases)} phrases...")
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i in range(0, len(sorted_phrases), batch_size):
                batch = sorted_phrases[i:i+batch_size]
                futures.append(executor.submit(process_batch, batch))
            for _ in tqdm(as_completed(futures), total=len(futures), desc="AI merging batches"):
                pass
        return merged_phrases

    def save_phrases(self, phrases, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("Welab Bank Negative - Aggressively Merged Phrases\n")
            f.write("="*50 + "\n\n")
            for phrase, freq in sorted(phrases.items(), key=lambda x: x[1], reverse=True):
                f.write(f"'{phrase}' (frequency: {freq})\n")

    def generate_wordcloud(self, phrases, filename):
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            c = self.negative_color
            return f"rgb({c[0]}, {c[1]}, {c[2]})"
        max_words = min(120, len(phrases))
        min_font_size = 16
        max_font_size = 160
        wordcloud = WordCloud(
            width=1200,
            height=1200,
            background_color='white',
            max_words=max_words,
            relative_scaling=0.9,
            collocations=False,
            prefer_horizontal=0.8,
            min_font_size=min_font_size,
            max_font_size=max_font_size,
            color_func=color_func
        ).generate_from_frequencies(phrases)
        plt.figure(figsize=(14, 14))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    api_key = DEEPSEEK_API_KEY
    processor = WelabNegativeAggressiveMerge(api_key)
    input_file = 'processed_phrases/welab_bank_negative_balanced_content_processed_phrases.txt'
    output_file = 'processed_phrases/welab_bank_negative_aggressive_merged_phrases.txt'
    wordcloud_file = 'wordclouds/welab_bank_negative_aggressive_merged_wordcloud.png'
    phrases = processor.parse_phrases(input_file)
    merged_phrases = processor.ai_aggressive_merge(phrases, target_count=180)
    processor.save_phrases(merged_phrases, output_file)
    processor.generate_wordcloud(merged_phrases, wordcloud_file)
    print(f"Aggressive merging complete! Output: {output_file}, Wordcloud: {wordcloud_file}")