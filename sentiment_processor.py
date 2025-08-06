#!/usr/bin/env python3
"""
Sentiment Phrase Processor
Processes sentiment phrases from multiple banks, translates non-English phrases,
merges similar phrases, and generates word clouds.
"""

import os
import re
import json
import requests
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from difflib import SequenceMatcher
import time
import logging
from config import *

# Set up logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class SentimentProcessor:
    def __init__(self, deepseek_api_key: str):
        self.deepseek_api_key = deepseek_api_key
        self.api_base_url = DEEPSEEK_API_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
    def is_english(self, text: str) -> bool:
        """Check if text is primarily English."""
        # Remove common punctuation and numbers
        cleaned = re.sub(r'[^\w\s]', '', text.lower())
        # Check if it contains mostly ASCII characters
        ascii_ratio = sum(1 for c in cleaned if ord(c) < 128) / len(cleaned) if cleaned else 0
        return ascii_ratio > ENGLISH_DETECTION_THRESHOLD
    
    def translate_to_english(self, text: str) -> str:
        """Translate non-English text to English using DeepSeek API."""
        if self.is_english(text):
            return text
            
        try:
            prompt = f"""Translate the following text to English. If it's already in English, return it as is. 
            Only return the translated text, nothing else.
            
            Text: "{text}"
            
            Translation:"""
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": TRANSLATION_MAX_TOKENS,
                "temperature": TRANSLATION_TEMPERATURE
            }
            
            response = requests.post(self.api_base_url, headers=self.headers, json=payload, timeout=API_TIMEOUT)
            response.raise_for_status()
            
            result = response.json()
            translated = result['choices'][0]['message']['content'].strip()
            
            # Clean up the response
            translated = re.sub(r'^["\']|["\']$', '', translated)
            
            logger.info(f"Translated: '{text}' -> '{translated}'")
            return translated
            
        except Exception as e:
            logger.error(f"Translation failed for '{text}': {e}")
            return text  # Return original if translation fails
    
    def similarity_score(self, phrase1: str, phrase2: str) -> float:
        """Calculate similarity between two phrases."""
        # Use SequenceMatcher for string similarity
        return SequenceMatcher(None, phrase1.lower(), phrase2.lower()).ratio()
    
    def merge_similar_phrases(self, phrases: Dict[str, int], similarity_threshold: float = SIMILARITY_THRESHOLD) -> Dict[str, int]:
        """Merge phrases with similar meanings and add up their frequencies."""
        if not phrases:
            return phrases
            
        # Sort phrases by frequency (descending) to prioritize higher frequency phrases
        sorted_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
        
        merged = {}
        used_indices = set()
        
        for i, (phrase1, freq1) in enumerate(sorted_phrases):
            if i in used_indices:
                continue
                
            total_freq = freq1
            best_phrase = phrase1
            
            # Check against all other phrases
            for j, (phrase2, freq2) in enumerate(sorted_phrases[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                if self.similarity_score(phrase1, phrase2) >= similarity_threshold:
                    total_freq += freq2
                    used_indices.add(j)
                    
                    # Keep the shorter phrase if frequencies are equal, otherwise keep the higher frequency one
                    if total_freq == freq1 + freq2:  # First merge
                        if MERGE_PREFER_SHORTER and len(phrase2) < len(best_phrase):
                            best_phrase = phrase2
                    elif freq2 > freq1:
                        best_phrase = phrase2
            
            merged[best_phrase] = total_freq
            used_indices.add(i)
        
        logger.info(f"Merged {len(phrases)} phrases into {len(merged)} phrases")
        return merged
    
    def parse_phrase_file(self, filepath: str) -> Dict[str, int]:
        """Parse a phrase file and extract phrases with their frequencies."""
        phrases = {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extract phrases using regex
            pattern = r"'([^']+)' \(frequency: (\d+)\)"
            matches = re.findall(pattern, content)
            
            for phrase, freq in matches:
                phrases[phrase] = int(freq)
                
            logger.info(f"Parsed {len(phrases)} phrases from {filepath}")
            return phrases
            
        except Exception as e:
            logger.error(f"Error parsing {filepath}: {e}")
            return {}
    
    def process_all_files(self, similarity_threshold: float = SIMILARITY_THRESHOLD) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Process all sentiment phrase files."""
        banks = BANKS
        sentiments = SENTIMENTS
        
        results = {}
        
        for bank in banks:
            results[bank] = {}
            
            for sentiment in sentiments:
                filename = f"{bank}_{sentiment}_phrases.txt"
                filepath = os.path.join('.', filename)
                
                if not os.path.exists(filepath):
                    logger.warning(f"File not found: {filepath}")
                    continue
                
                logger.info(f"Processing {filename}...")
                
                # Parse phrases
                phrases = self.parse_phrase_file(filepath)
                
                if not phrases:
                    continue
                
                # Translate non-English phrases
                translated_phrases = {}
                for phrase, freq in phrases.items():
                    translated = self.translate_to_english(phrase)
                    translated_phrases[translated] = translated_phrases.get(translated, 0) + freq
                
                # Merge similar phrases
                merged_phrases = self.merge_similar_phrases(translated_phrases, similarity_threshold)
                
                results[bank][sentiment] = merged_phrases
                
                # Add delay to avoid API rate limits
                time.sleep(API_RETRY_DELAY)
        
        return results
    
    def generate_wordcloud(self, phrases: Dict[str, int], title: str, filename: str):
        """Generate a word cloud from phrases."""
        if not phrases:
            logger.warning(f"No phrases to generate wordcloud for {title}")
            return
        
        # Create word cloud
        wordcloud = WordCloud(
            width=WORDCLOUD_WIDTH,
            height=WORDCLOUD_HEIGHT,
            background_color='white',
            max_words=WORDCLOUD_MAX_WORDS,
            colormap=WORDCLOUD_COLORMAP,
            relative_scaling=WORDCLOUD_RELATIVE_SCALING
        ).generate_from_frequencies(phrases)
        
        # Create plot
        plt.figure(figsize=(15, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=20, pad=20)
        plt.tight_layout()
        
        # Save the wordcloud
        plt.savefig(filename, dpi=WORDCLOUD_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated wordcloud: {filename}")
    
    def generate_all_wordclouds(self, results: Dict[str, Dict[str, Dict[str, int]]]):
        """Generate word clouds for all banks and sentiments."""
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        for bank, sentiments in results.items():
            for sentiment, phrases in sentiments.items():
                if phrases:
                    title = f"{bank.replace('_', ' ').title()} - {sentiment.title()} Phrases"
                    filename = f"{OUTPUT_DIR}/{bank}_{sentiment}_wordcloud.png"
                    self.generate_wordcloud(phrases, title, filename)
    
    def save_processed_results(self, results: Dict[str, Dict[str, Dict[str, int]]], filename: str = PROCESSED_RESULTS_FILE):
        """Save processed results to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved processed results to {filename}")
    
    def print_summary(self, results: Dict[str, Dict[str, Dict[str, int]]]):
        """Print a summary of the processing results."""
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        
        for bank, sentiments in results.items():
            print(f"\n{bank.replace('_', ' ').title()}:")
            for sentiment, phrases in sentiments.items():
                total_freq = sum(phrases.values())
                print(f"  {sentiment.title()}: {len(phrases)} unique phrases, {total_freq} total frequency")
                
                # Show top 5 phrases
                top_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)[:5]
                print("    Top phrases:")
                for phrase, freq in top_phrases:
                    print(f"      '{phrase}' (frequency: {freq})")

def main():
    """Main function to run the sentiment processor."""
    # Get API key from config or user input
    api_key = DEEPSEEK_API_KEY
    
    if not api_key:
        api_key = input("Please enter your DeepSeek API key: ").strip()
        
    if not api_key:
        print("Error: API key is required")
        print("You can either:")
        print("1. Add your API key to the DEEPSEEK_API_KEY variable in config.py")
        print("2. Enter it when prompted")
        return
    
    # Initialize processor
    processor = SentimentProcessor(api_key)
    
    # Process all files
    print("Processing sentiment phrases...")
    results = processor.process_all_files(similarity_threshold=SIMILARITY_THRESHOLD)
    
    # Save results
    processor.save_processed_results(results)
    
    # Generate word clouds
    print("Generating word clouds...")
    processor.generate_all_wordclouds(results)
    
    # Print summary
    processor.print_summary(results)
    
    print("\nProcessing complete! Check the 'wordclouds' directory for generated images.")

if __name__ == "__main__":
    main() 