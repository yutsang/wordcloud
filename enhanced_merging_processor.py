#!/usr/bin/env python3
"""
Enhanced Merging Processor
- More aggressive merging to reduce phrase count
- Remove generic words (i, me, those, etc.)
- Remove meaningless phrases
- Make text larger in word clouds
- Process all 3 banks (Mox, Welab, ZA)
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

class EnhancedMergingProcessor:
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
        
        # Enhanced stop words and generic phrases
        self.generic_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'will', 
            'would', 'could', 'should', 'ought', 'all', 'any', 'both', 'each', 'few', 'more', 
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
            'than', 'too', 'very', 'can', 'just', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'
        }
        
        self.meaningless_phrases = {
            'matter how many', 'how many times', 'google play store', 'play store', 'app store',
            'matter how', 'how many', 'many times', 'times but', 'but still', 'still cannot',
            'cannot open', 'open the', 'the app', 'app is', 'is not', 'not working', 'working properly',
            'properly and', 'and still', 'still trying', 'trying to', 'to get', 'get past',
            'past the', 'the verification', 'verification process', 'process is', 'is too',
            'too complicated', 'complicated and', 'and frustrating', 'frustrating to', 'to use',
            'use the', 'the interface', 'interface is', 'is confusing', 'confusing and', 'and difficult',
            'difficult to', 'to navigate', 'navigate through', 'through the', 'the app',
            'app crashes', 'crashes when', 'when i', 'i try', 'try to', 'to login', 'login to',
            'to my', 'my account', 'account and', 'and it', 'it keeps', 'keeps crashing',
            'crashing every', 'every time', 'time i', 'i open', 'open it', 'it up',
            'up and', 'and down', 'down all', 'all the', 'the time', 'time and', 'and it',
            'it is', 'is very', 'very slow', 'slow and', 'and unresponsive', 'unresponsive most',
            'most of', 'of the', 'the time', 'time so', 'so frustrating', 'frustrating and',
            'and annoying', 'annoying to', 'to use', 'use this', 'this app', 'app anymore',
            'anymore because', 'because it', 'it is', 'is just', 'just too', 'too buggy',
            'buggy and', 'and unreliable', 'unreliable for', 'for daily', 'daily use',
            'use and', 'and i', 'i would', 'would not', 'not recommend', 'recommend it',
            'it to', 'to anyone', 'anyone who', 'who wants', 'wants a', 'a reliable',
            'reliable banking', 'banking app', 'app that', 'that works', 'works properly',
            'properly without', 'without constant', 'constant crashes', 'crashes and', 'and errors',
            'if you', 'you can', 'can not', 'not able', 'able to', 'to do', 'do this',
            'this is', 'is a', 'a very', 'very bad', 'bad app', 'app and', 'and i',
            'i have', 'have been', 'been trying', 'trying for', 'for days', 'days and',
            'and still', 'still cannot', 'cannot get', 'get it', 'it to', 'to work',
            'work properly', 'properly so', 'so i', 'i am', 'am giving', 'giving up',
            'up on', 'on this', 'this app', 'app completely', 'completely and',
            'and will', 'will use', 'use a', 'a different', 'different bank', 'bank instead'
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
    
    def remove_generic_words(self, phrase: str) -> str:
        """Remove generic words from phrases."""
        words = phrase.split()
        filtered_words = [word for word in words if word.lower() not in self.generic_words]
        return ' '.join(filtered_words).strip()
    
    def is_meaningless_phrase(self, phrase: str) -> bool:
        """Check if a phrase is meaningless."""
        phrase_lower = phrase.lower().strip()
        
        # Check for meaningless patterns
        for meaningless_pattern in self.meaningless_phrases:
            if meaningless_pattern in phrase_lower:
                return True
        
        # Check for generic patterns
        if re.search(r'\b(i|me|my|you|your|he|she|it|they|them|this|that|these|those)\b', phrase_lower):
            if len(phrase_lower.split()) <= 3:  # Short phrases with pronouns are likely meaningless
                return True
        
        # Check for repetitive patterns
        if re.search(r'\b(very|really|so|too|extremely)\s+(bad|terrible|awful|horrible|worst|poor|disappointing|frustrating|annoying|difficult|hard|complicated|confusing|slow|broken|not working|does not work|doesn\'t work)\b', phrase_lower):
            return True
        
        return False
    
    def aggressive_filtering(self, phrases: Dict[str, int]) -> Dict[str, int]:
        """Aggressive filtering to remove generic and meaningless phrases."""
        filtered_phrases = {}
        
        print("Aggressive filtering to remove generic and meaningless phrases...")
        with tqdm(total=len(phrases), desc="Filtering phrases") as pbar:
            for phrase, freq in phrases.items():
                # Remove generic words
                cleaned_phrase = self.remove_generic_words(phrase)
                
                # Skip if phrase becomes too short after cleaning
                if len(cleaned_phrase.strip()) < 3:
                    pbar.update(1)
                    continue
                
                # Check if phrase is meaningless
                if not self.is_meaningless_phrase(cleaned_phrase):
                    # Additional filtering criteria
                    if len(cleaned_phrase.strip()) >= 4 and len(cleaned_phrase.strip()) <= 50:
                        # Check for stop words dominance
                        words = cleaned_phrase.lower().split()
                        if len(words) >= 2:
                            stop_word_count = sum(1 for word in words if word in self.generic_words)
                            if stop_word_count / len(words) <= 0.3:  # Very strict stop word ratio
                                filtered_phrases[cleaned_phrase] = freq
                pbar.update(1)
        
        logger.info(f"Aggressively filtered {len(phrases)} phrases to {len(filtered_phrases)} phrases")
        return filtered_phrases
    
    def ai_aggressive_merging(self, phrases: Dict[str, int], sentiment: str) -> Dict[str, int]:
        """Use AI to aggressively merge phrases."""
        if len(phrases) <= 1:
            return phrases
        
        # Sort phrases by frequency
        sorted_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
        
        # Take only top phrases to reduce processing time
        top_phrases = sorted_phrases[:400]  # Limit to top 400 phrases
        
        # Group phrases into batches
        batch_size = 20
        merged_phrases = {}
        
        def process_merging_batch(batch):
            batch_merged = {}
            
            if len(batch) == 1:
                phrase, freq = batch[0]
                if phrase.strip():
                    batch_merged[phrase] = freq
                return batch_merged
            
            # Create prompt for aggressive AI merging
            phrase_list = [f"'{phrase}' (frequency: {freq})" for phrase, freq in batch]
            phrases_text = "\n".join(phrase_list)
            
            prompt = f"""Analyze these {sentiment} sentiment phrases about banking apps and AGGRESSIVELY merge those with similar meanings. 
            Be very aggressive in merging - combine phrases that are even remotely similar in meaning.
            Choose the most representative phrase for each group and sum up all frequencies.
            Reduce the number of phrases by at least 60-70%.
            
            Phrases:
            {phrases_text}
            
            Return the result as a JSON array of objects with 'phrase' and 'frequency' fields.
            Be extremely aggressive in merging similar phrases.
            
            Example format:
            [
                {{"phrase": "merged phrase", "frequency": total_frequency}},
                {{"phrase": "separate phrase", "frequency": original_frequency}}
            ]
            
            Result:"""
            
            try:
                payload = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.3
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
        
        print(f"AI aggressive merging {len(top_phrases)} {sentiment} phrases with {self.num_workers} workers...")
        
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
        """Process a single bank file with enhanced merging."""
        filename = f"{bank}_{sentiment}_phrases.txt"
        filepath = os.path.join('.', filename)
        
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return {}
        
        print(f"\nProcessing {filename} with enhanced merging...")
        
        # Parse phrases
        phrases = self.parse_phrase_file(filepath)
        
        if not phrases:
            return {}
        
        # Round 1: Aggressive filtering
        print("Round 1: Aggressive filtering...")
        filtered_phrases = self.aggressive_filtering(phrases)
        
        # Round 2: AI aggressive merging
        print("Round 2: AI aggressive merging...")
        merged_phrases = self.ai_aggressive_merging(filtered_phrases, sentiment)
        
        # Round 3: Final merging if still too many phrases
        if len(merged_phrases) > 200:
            print("Round 3: Final AI merging...")
            merged_phrases = self.ai_aggressive_merging(merged_phrases, sentiment)
        
        # Save processed phrases to text file
        processed_filename = f"{bank}_{sentiment}_enhanced_merging_processed_phrases.txt"
        self.save_processed_phrases_to_txt(merged_phrases, processed_filename)
        
        return merged_phrases
    
    def save_processed_phrases_to_txt(self, phrases: Dict[str, int], filename: str):
        """Save processed phrases to a text file."""
        os.makedirs('processed_phrases', exist_ok=True)
        filepath = os.path.join('processed_phrases', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Enhanced Merging - Processed Phrases\n")
            f.write("="*50 + "\n\n")
            
            # Sort by frequency (descending)
            sorted_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
            
            for phrase, freq in sorted_phrases:
                f.write(f"'{phrase}' (frequency: {freq})\n")
        
        logger.info(f"Saved processed phrases to {filepath}")
    
    def process_all_banks(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Process all banks with enhanced merging."""
        banks = ['mox_bank', 'welab_bank', 'za_bank']
        sentiments = ['negative']  # Focus on negative as requested
        
        results = {}
        
        print(f"Processing all banks with enhanced merging...")
        
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
        """Generate a word cloud with larger text."""
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
        
        # Adjust word cloud parameters for larger text
        max_words = min(150, len(phrases))  # Fewer words for larger text
        min_font_size = 15  # Larger minimum font size
        max_font_size = 150  # Larger maximum font size
        
        wordcloud = WordCloud(
            width=1200,
            height=1200,
            background_color='white',
            max_words=max_words,
            relative_scaling=0.8,  # Better word size distribution
            collocations=False,
            prefer_horizontal=0.7,  # More horizontal text
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
        
        print("\nGenerating enhanced merging word clouds...")
        
        for bank, sentiments in results.items():
            for sentiment, phrases in sentiments.items():
                if phrases:
                    filename = f"{OUTPUT_DIR}/{bank}_{sentiment}_enhanced_merging_wordcloud.png"
                    self.generate_wordcloud(phrases, sentiment, bank, filename)
    
    def print_summary(self, results: Dict[str, Dict[str, Dict[str, int]]]):
        """Print a summary of the processing results."""
        print("\n" + "="*60)
        print("ENHANCED MERGING PROCESSING SUMMARY")
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
    """Main function to run the enhanced merging processor."""
    api_key = DEEPSEEK_API_KEY
    
    if not api_key:
        api_key = input("Please enter your DeepSeek API key: ").strip()
        
    if not api_key:
        print("Error: API key is required")
        return
    
    print("Enhanced Merging Processor")
    print("="*50)
    print(f"Positive color: RGB{EnhancedMergingProcessor(api_key).positive_color}")
    print(f"Negative color: RGB{EnhancedMergingProcessor(api_key).negative_color}")
    print(f"Backing up existing wordclouds...")
    
    processor = EnhancedMergingProcessor(api_key, num_workers=15)
    
    # Backup existing wordclouds
    backup_dir = processor.backup_existing_wordclouds()
    
    print("Processing all banks with enhanced merging...")
    start_time = time.time()
    
    results = processor.process_all_banks()
    
    print("Generating enhanced merging word clouds...")
    processor.generate_all_wordclouds(results)
    
    processor.print_summary(results)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\nEnhanced merging processing complete!")
    print(f"Total processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)")
    print(f"Backup location: {backup_dir}")
    print(f"Processed phrases saved to 'processed_phrases/' directory")
    print(f"Check the '{OUTPUT_DIR}' directory for the generated wordclouds.")

if __name__ == "__main__":
    main() 