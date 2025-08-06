#!/usr/bin/env python3
"""
Final Cleanup Processor
- Fix incomplete phrases (add missing words)
- Merge redundant phrases
- Remove mixed sentiment issues
- Clean up technical jargon
- Process all 3 banks for final cleanup
"""

import os
import re
import json
import requests
import shutil
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import time
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import *

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class FinalCleanupProcessor:
    def __init__(self, deepseek_api_key: str, num_workers: int = 15):
        self.deepseek_api_key = deepseek_api_key
        self.api_base_url = DEEPSEEK_API_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {deepseek_api_key}",
            "Content-Type": "application/json"
        }
        self.num_workers = num_workers
        
        # Create session with connection pooling and retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=num_workers, pool_maxsize=num_workers)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Custom colors
        self.positive_color = (12, 35, 60)
        self.negative_color = (0, 184, 245)
        
        # Phrases to fix/merge
        self.phrase_fixes = {
            'intentionally difficult understand': 'intentionally difficult to understand',
            'unable login': 'unable to login',
            'cannot login': 'cannot login',
            'one worst': 'one of the worst',
            'slower slower': 'getting slower',
            'becoming slower slower': 'performance getting worse',
            'cannot screen cap': 'screenshot restrictions',
            'cannot screen': 'screenshot restrictions',
            'screen cap': 'screenshot restrictions',
            'from mox': 'mox bank issues',
            'poor customer service': 'customer service problems',
            'terrible customer service': 'customer service problems',
            'unhelpful customer': 'customer service problems',
            'customer service horrible': 'customer service problems',
            'qr code': 'qr code scanning issues',
            'android 8': 'android compatibility issues',
            'works well': None,  # Remove positive phrases from negative context
            'very good': None,
            'good experience': None,
            'user friendly': None,
            'easy to': None,
            'convenient': None
        }
        
        # Technical jargon to remove or simplify
        self.technical_jargon = {
            'android 8', 'android 9', 'android 10', 'android 11', 'android 12',
            'ios 14', 'ios 15', 'ios 16', 'ios 17',
            'samsung fold', 'huawei y7', 'y7 prime 2018',
            'system error', 'error code', 'error message',
            'api', 'sdk', 'framework', 'backend', 'frontend'
        }
    
    def backup_existing_wordclouds(self):
        """Backup existing wordclouds before generating new ones."""
        backup_dir = os.path.join(OUTPUT_DIR, 'backup')
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_subdir = os.path.join(backup_dir, f"backup_{timestamp}")
        os.makedirs(backup_subdir, exist_ok=True)
        
        if os.path.exists(OUTPUT_DIR):
            for file in os.listdir(OUTPUT_DIR):
                if file.endswith('.png') and 'wordcloud' in file and ('negative' in file or 'positive' in file):
                    src = os.path.join(OUTPUT_DIR, file)
                    dst = os.path.join(backup_subdir, file)
                    shutil.copy2(src, dst)
                    logger.info(f"Backed up: {file}")
        
        logger.info(f"Backup completed: {backup_subdir}")
        return backup_subdir
    
    def parse_phrase_file(self, filepath: str) -> Dict[str, int]:
        """Parse a phrase file and extract phrases with their frequencies."""
        phrases = {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            pattern = r"'([^']+)' \(frequency: (\d+)\)"
            matches = re.findall(pattern, content)
            
            for phrase, freq in matches:
                phrases[phrase] = int(freq)
                
            logger.info(f"Parsed {len(phrases)} phrases from {filepath}")
            return phrases
            
        except Exception as e:
            logger.error(f"Error parsing {filepath}: {e}")
            return {}
    
    def cleanup_phrases(self, phrases: Dict[str, int], sentiment: str) -> Dict[str, int]:
        """Clean up phrases by fixing incomplete phrases and removing issues."""
        cleaned_phrases = {}
        
        print("Cleaning up phrases...")
        with tqdm(total=len(phrases), desc="Cleaning phrases") as pbar:
            for phrase, freq in phrases.items():
                # Skip phrases that should be removed
                if phrase in self.phrase_fixes and self.phrase_fixes[phrase] is None:
                    pbar.update(1)
                    continue
                
                # Fix incomplete phrases
                if phrase in self.phrase_fixes:
                    fixed_phrase = self.phrase_fixes[phrase]
                    cleaned_phrases[fixed_phrase] = cleaned_phrases.get(fixed_phrase, 0) + freq
                else:
                    # Check for technical jargon
                    is_technical = False
                    for jargon in self.technical_jargon:
                        if jargon.lower() in phrase.lower():
                            is_technical = True
                            break
                    
                    if not is_technical:
                        cleaned_phrases[phrase] = freq
                
                pbar.update(1)
        
        logger.info(f"Cleaned {len(phrases)} phrases to {len(cleaned_phrases)} phrases")
        return cleaned_phrases
    
    def ai_final_merging(self, phrases: Dict[str, int], sentiment: str) -> Dict[str, int]:
        """Use AI for final merging to create the most meaningful phrases."""
        if len(phrases) <= 1:
            return phrases
        
        # Sort phrases by frequency
        sorted_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
        
        # Take only top phrases to reduce processing time
        top_phrases = sorted_phrases[:300]  # Limit to top 300 phrases
        
        # Group phrases into batches
        batch_size = 15
        merged_phrases = {}
        
        def process_merging_batch(batch):
            batch_merged = {}
            
            if len(batch) == 1:
                phrase, freq = batch[0]
                if phrase.strip():
                    batch_merged[phrase] = freq
                return batch_merged
            
            # Create prompt for final AI merging
            phrase_list = [f"'{phrase}' (frequency: {freq})" for phrase, freq in batch]
            phrases_text = "\n".join(phrase_list)
            
            prompt = f"""Analyze these {sentiment} sentiment phrases about banking apps and create the most meaningful, clear, and actionable phrases.
            
            Rules:
            1. Merge similar phrases into the most representative one
            2. Fix any incomplete or unclear phrases
            3. Choose the most descriptive and actionable phrase for each group
            4. Remove any remaining generic or unclear phrases
            5. Focus on phrases that clearly indicate specific problems or issues
            6. Reduce the number of phrases by 50-60%
            
            Phrases:
            {phrases_text}
            
            Return the result as a JSON array of objects with 'phrase' and 'frequency' fields.
            Only include the most meaningful and actionable phrases.
            
            Example format:
            [
                {{"phrase": "meaningful phrase", "frequency": total_frequency}},
                {{"phrase": "another meaningful phrase", "frequency": total_frequency}}
            ]
            
            Result:"""
            
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
                
                # Try to parse JSON response
                try:
                    merged_batch = json.loads(ai_response)
                    for item in merged_batch:
                        if 'phrase' in item and 'frequency' in item:
                            phrase = item['phrase'].strip()
                            if phrase and len(phrase) >= 3:
                                batch_merged[phrase] = batch_merged.get(phrase, 0) + item['frequency']
                except json.JSONDecodeError:
                    # If JSON parsing fails, keep original phrases
                    for phrase, freq in batch:
                        if phrase.strip():
                            batch_merged[phrase] = batch_merged.get(phrase, 0) + freq
                
            except Exception as e:
                logger.error(f"AI merging failed for batch: {e}")
                # Keep original phrases if AI fails
                for phrase, freq in batch:
                    if phrase.strip():
                        batch_merged[phrase] = batch_merged.get(phrase, 0) + freq
            
            return batch_merged
        
        print(f"AI final merging {len(top_phrases)} {sentiment} phrases with {self.num_workers} workers...")
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i in range(0, len(top_phrases), batch_size):
                batch = top_phrases[i:i + batch_size]
                futures.append(executor.submit(process_merging_batch, batch))
            
            with tqdm(total=len(futures), desc="AI merging batches") as pbar:
                for future in as_completed(futures):
                    try:
                        batch_result = future.result()
                        for phrase, freq in batch_result.items():
                            merged_phrases[phrase] = merged_phrases.get(phrase, 0) + freq
                    except Exception as e:
                        logger.error(f"Error in batch processing: {e}")
                    pbar.update(1)
                    time.sleep(0.1)
        
        return merged_phrases
    
    def process_bank_file(self, bank: str, sentiment: str) -> Dict[str, int]:
        """Process a single bank file with final cleanup."""
        filename = f"{bank}_{sentiment}_enhanced_merging_processed_phrases.txt"
        filepath = os.path.join('processed_phrases', filename)
        
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return {}
        
        print(f"\nProcessing {filename} with final cleanup...")
        
        # Parse phrases
        phrases = self.parse_phrase_file(filepath)
        
        if not phrases:
            return {}
        
        # Round 1: Clean up phrases
        print("Round 1: Cleaning up phrases...")
        cleaned_phrases = self.cleanup_phrases(phrases, sentiment)
        
        # Round 2: AI final merging
        print("Round 2: AI final merging...")
        final_phrases = self.ai_final_merging(cleaned_phrases, sentiment)
        
        # Save processed phrases to text file
        processed_filename = f"{bank}_{sentiment}_final_cleanup_processed_phrases.txt"
        self.save_processed_phrases_to_txt(final_phrases, processed_filename)
        
        return final_phrases
    
    def save_processed_phrases_to_txt(self, phrases: Dict[str, int], filename: str):
        """Save processed phrases to a text file."""
        os.makedirs('processed_phrases', exist_ok=True)
        filepath = os.path.join('processed_phrases', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Final Cleanup - Processed Phrases\n")
            f.write("="*50 + "\n\n")
            
            # Sort by frequency (descending)
            sorted_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
            
            for phrase, freq in sorted_phrases:
                f.write(f"'{phrase}' (frequency: {freq})\n")
        
        logger.info(f"Saved processed phrases to {filepath}")
    
    def process_all_banks(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Process all banks with final cleanup."""
        banks = ['mox_bank', 'welab_bank', 'za_bank']
        sentiments = ['negative']  # Focus on negative as requested
        
        results = {}
        
        print(f"Processing all banks with final cleanup...")
        
        for bank in banks:
            results[bank] = {}
            
            for sentiment in sentiments:
                print(f"\n{'='*60}")
                print(f"Processing {bank} {sentiment}...")
                print(f"{'='*60}")
                
                start_time = time.time()
                
                bank_results = self.process_bank_file(bank, sentiment)
                results[bank][sentiment] = bank_results
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                print(f"\n{bank} {sentiment} processing complete!")
                print(f"Processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)")
                print(f"Results: {len(bank_results)} unique phrases")
                
                time.sleep(2)
        
        return results
    
    def generate_wordcloud(self, phrases: Dict[str, int], sentiment: str, bank: str, filename: str):
        """Generate a word cloud with optimal text size."""
        if not phrases:
            logger.warning(f"No phrases to generate wordcloud for {filename}")
            return
        
        # Choose color based on sentiment
        if sentiment == 'positive':
            color = self.positive_color
        else:
            color = self.negative_color
        
        # Create custom color function
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            return f"rgb({color[0]}, {color[1]}, {color[2]})"
        
        # Adjust word cloud parameters for optimal readability
        max_words = min(120, len(phrases))  # Fewer words for better readability
        min_font_size = 18  # Larger minimum font size
        max_font_size = 160  # Larger maximum font size
        
        wordcloud = WordCloud(
            width=1200,
            height=1200,
            background_color='white',
            max_words=max_words,
            relative_scaling=0.9,  # Better word size distribution
            collocations=False,
            prefer_horizontal=0.8,  # More horizontal text for readability
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
        
        logger.info(f"Generated wordcloud: {filename}")
    
    def generate_all_wordclouds(self, results: Dict[str, Dict[str, Dict[str, int]]]):
        """Generate word clouds for all banks and sentiments."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        print("\nGenerating final cleanup word clouds...")
        
        for bank, sentiments in results.items():
            for sentiment, phrases in sentiments.items():
                if phrases:
                    filename = f"{OUTPUT_DIR}/{bank}_{sentiment}_final_cleanup_wordcloud.png"
                    self.generate_wordcloud(phrases, sentiment, bank, filename)
    
    def print_summary(self, results: Dict[str, Dict[str, Dict[str, int]]]):
        """Print a summary of the processing results."""
        print("\n" + "="*60)
        print("FINAL CLEANUP PROCESSING SUMMARY")
        print("="*60)
        
        for bank, sentiments in results.items():
            print(f"\n{bank.replace('_', ' ').title()}:")
            for sentiment, phrases in sentiments.items():
                total_freq = sum(phrases.values())
                print(f"  {sentiment.title()}: {len(phrases)} unique phrases, {total_freq} total frequency")
                
                # Show top 10 phrases
                top_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)[:10]
                print("    Top phrases:")
                for i, (phrase, freq) in enumerate(top_phrases, 1):
                    print(f"      {i:2d}. '{phrase}' (frequency: {freq})")

def main():
    """Main function to run the final cleanup processor."""
    api_key = DEEPSEEK_API_KEY
    
    if not api_key:
        api_key = input("Please enter your DeepSeek API key: ").strip()
        
    if not api_key:
        print("Error: API key is required")
        return
    
    print("Final Cleanup Processor")
    print("="*50)
    print(f"Positive color: RGB{FinalCleanupProcessor(api_key).positive_color}")
    print(f"Negative color: RGB{FinalCleanupProcessor(api_key).negative_color}")
    print(f"Backing up existing wordclouds...")
    
    processor = FinalCleanupProcessor(api_key, num_workers=15)
    
    # Backup existing wordclouds
    backup_dir = processor.backup_existing_wordclouds()
    
    print("Processing all banks with final cleanup...")
    start_time = time.time()
    
    results = processor.process_all_banks()
    
    print("Generating final cleanup word clouds...")
    processor.generate_all_wordclouds(results)
    
    processor.print_summary(results)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\nFinal cleanup processing complete!")
    print(f"Total processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)")
    print(f"Backup location: {backup_dir}")
    print(f"Processed phrases saved to 'processed_phrases/' directory")
    print(f"Check the '{OUTPUT_DIR}' directory for the generated wordclouds.")

if __name__ == "__main__":
    main() 