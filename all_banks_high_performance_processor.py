#!/usr/bin/env python3
"""
All Banks High Performance Processor
Features:
- Processes Welab and ZA banks (positive and negative)
- High performance with 15-20 workers
- Filters out unclear phrases like "matter how many", "how many times", "google play store"
- Optimized API handling with connection pooling
- Aggressive merging to reduce phrase count
- Better text ratio balancing
- Custom colors for word clouds (positive: RGB(12,35,60), negative: RGB(0,184,245))
- Square word clouds without titles
- Saves processed text files
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

# Set up logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class AllBanksHighPerformanceProcessor:
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
        
        # Custom colors for word clouds
        self.positive_color = (12, 35, 60)  # RGB for positive
        self.negative_color = (0, 184, 245)  # RGB for negative
        
        # Comprehensive stop words
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
            'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
            'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'having', 'do', 'does', 'did', 'doing', 'would', 'could', 'should', 'ought',
            'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just',
            'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn',
            'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn',
            'weren', 'won', 'wouldn', 'im', 'youre', 'hes', 'shes', 'its', 'were', 'theyre',
            'ive', 'youve', 'weve', 'theyve', 'id', 'youd', 'hed', 'shed', 'wed', 'theyd',
            'ill', 'youll', 'hell', 'shell', 'well', 'theyll', 'isnt', 'arent', 'wasnt',
            'weren', 'hasnt', 'havent', 'hadnt', 'doesnt', 'dont', 'didnt', 'wont', 'wouldnt',
            'couldnt', 'shouldnt', 'lets', 'thats', 'whos', 'whats', 'heres', 'theres', 'whens',
            'wheres', 'whys', 'hows', 'us', 'him', 'her', 'them', 'their', 'ours', 'yours',
            'mine', 'yours', 'his', 'hers', 'theirs', 'myself', 'yourself', 'himself', 'herself',
            'itself', 'ourselves', 'yourselves', 'themselves'
        }
        
        # Unclear phrases to filter out
        self.unclear_phrases = {
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
            'properly without', 'without constant', 'constant crashes', 'crashes and', 'and errors'
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
                if file.endswith('.png') and 'wordcloud' in file and ('welab' in file or 'za_bank' in file):
                    src = os.path.join(OUTPUT_DIR, file)
                    dst = os.path.join(backup_subdir, file)
                    shutil.copy2(src, dst)
                    logger.info(f"Backed up: {file}")
        
        logger.info(f"Backup completed: {backup_subdir}")
        return backup_subdir
    
    def is_english(self, text: str) -> bool:
        """Check if text is primarily English."""
        cleaned = re.sub(r'[^\w\s]', '', text.lower())
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
            
            response = self.session.post(self.api_base_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            translated = result['choices'][0]['message']['content'].strip()
            translated = re.sub(r'^["\']|["\']$', '', translated)
            
            return translated
            
        except Exception as e:
            logger.error(f"Translation failed for '{text}': {e}")
            return text
    
    def translate_phrases_parallel(self, phrases: Dict[str, int]) -> Dict[str, int]:
        """Translate phrases using parallel processing with optimized workers."""
        if not phrases:
            return {}
        
        phrase_items = list(phrases.items())
        translated_phrases = {}
        
        def translate_single_phrase(item):
            phrase, freq = item
            translated = self.translate_to_english(phrase)
            return translated, freq
        
        print(f"Translating phrases with {self.num_workers} workers...")
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(translate_single_phrase, item) for item in phrase_items]
            
            with tqdm(total=len(futures), desc="Translating phrases") as pbar:
                for future in as_completed(futures):
                    try:
                        translated, freq = future.result()
                        translated_phrases[translated] = translated_phrases.get(translated, 0) + freq
                    except Exception as e:
                        logger.error(f"Error in translation: {e}")
                    pbar.update(1)
                    time.sleep(0.05)  # Small delay to avoid overwhelming API
        
        return translated_phrases
    
    def aggressive_filtering_with_unclear_removal(self, phrases: Dict[str, int]) -> Dict[str, int]:
        """Aggressive filtering to reduce phrase count and remove unclear phrases."""
        filtered_phrases = {}
        
        print("Aggressive filtering with unclear phrase removal...")
        with tqdm(total=len(phrases), desc="Filtering phrases") as pbar:
            for phrase, freq in phrases.items():
                # Check for unclear phrases
                phrase_lower = phrase.lower().strip()
                is_unclear = False
                
                # Check if phrase contains unclear patterns
                for unclear_pattern in self.unclear_phrases:
                    if unclear_pattern in phrase_lower:
                        is_unclear = True
                        break
                
                # Check for generic patterns
                if re.search(r'\b(matter|how|many|times|google|play|store|app|store)\b', phrase_lower):
                    if len(phrase_lower.split()) <= 3:  # Short phrases with these words are likely unclear
                        is_unclear = True
                
                if not is_unclear:
                    # More aggressive filtering
                    if len(phrase.strip()) >= 4 and len(phrase.strip()) <= 60:
                        # Check for stop words dominance
                        words = phrase.lower().split()
                        if len(words) >= 2:  # Only keep phrases with at least 2 words
                            stop_word_count = sum(1 for word in words if word in self.stop_words)
                            if stop_word_count / len(words) <= 0.5:  # More strict stop word ratio
                                # Remove phrases that are too generic
                                if not self.is_too_generic(phrase):
                                    filtered_phrases[phrase] = freq
                pbar.update(1)
        
        logger.info(f"Aggressively filtered {len(phrases)} phrases to {len(filtered_phrases)} phrases")
        return filtered_phrases
    
    def is_too_generic(self, phrase: str) -> bool:
        """Check if a phrase is too generic."""
        generic_patterns = [
            r'^\s*(bad|terrible|awful|horrible|worst|poor|disappointing|frustrating|annoying|difficult|hard|complicated|confusing|slow|broken|not working|does not work|doesn\'t work)\s*$',
            r'^\s*(very|really|so|too|extremely)\s+(bad|terrible|awful|horrible|worst|poor|disappointing|frustrating|annoying|difficult|hard|complicated|confusing|slow|broken)\s*$',
            r'^\s*(customer service|service|app|bank|banking|account|money|payment|transfer|transaction|login|password|verification|security|update|version|system|process|procedure|function|feature|interface|ui|ux|design|layout|screen|page|button|menu|option|setting|configuration|data|information|file|document|record|history|log|report|status|result|response|message|notification|alert|error|warning|success|failure|problem|issue|bug|glitch|crash|freeze|hang|slow|fast|speed|performance|efficiency|quality|reliability|stability|compatibility|accessibility|usability|functionality)\s*$'
        ]
        
        phrase_lower = phrase.lower().strip()
        for pattern in generic_patterns:
            if re.match(pattern, phrase_lower):
                return True
        return False
    
    def ai_screening_parallel(self, phrases: Dict[str, int], sentiment: str) -> Dict[str, int]:
        """Use AI to screen phrases with parallel processing."""
        if not phrases:
            return phrases
        
        # Sort phrases by frequency
        sorted_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
        
        # Take only top phrases to reduce processing time
        top_phrases = sorted_phrases[:600]  # Limit to top 600 phrases
        
        # Group phrases into batches
        batch_size = 20
        screened_phrases = {}
        
        def process_screening_batch(batch):
            batch_phrases = {}
            
            # Create prompt for AI screening
            phrase_list = [f"'{phrase}' (frequency: {freq})" for phrase, freq in batch]
            phrases_text = "\n".join(phrase_list)
            
            prompt = f"""Analyze these {sentiment} sentiment phrases about banking apps. 
            Keep ONLY the most meaningful, clear, and representative phrases.
            Remove phrases that are:
            1. Incomplete or fragmented
            2. Too generic or vague
            3. Technical jargon without clear meaning
            4. Repetitive or redundant
            5. Unclear or confusing
            6. Too short or too long
            7. Not specific enough
            8. Phrases like "matter how many", "how many times", "google play store", etc.
            
            Be very selective - keep only the best 30-40% of phrases.
            
            Phrases to analyze:
            {phrases_text}
            
            Return the result as a JSON array of objects with 'phrase' and 'frequency' fields.
            Only include the most meaningful and clear phrases.
            
            Example format:
            [
                {{"phrase": "meaningful phrase", "frequency": original_frequency}},
                {{"phrase": "another meaningful phrase", "frequency": original_frequency}}
            ]
            
            Result:"""
            
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
                
                # Try to parse JSON response
                try:
                    screened_batch = json.loads(ai_response)
                    for item in screened_batch:
                        if 'phrase' in item and 'frequency' in item:
                            phrase = item['phrase'].strip()
                            if phrase and len(phrase) >= 3:
                                batch_phrases[phrase] = item['frequency']
                except json.JSONDecodeError:
                    # If JSON parsing fails, keep original phrases
                    for phrase, freq in batch:
                        if phrase.strip():
                            batch_phrases[phrase] = freq
                
            except Exception as e:
                logger.error(f"AI screening failed for batch: {e}")
                # Keep original phrases if AI fails
                for phrase, freq in batch:
                    if phrase.strip():
                        batch_phrases[phrase] = freq
            
            return batch_phrases
        
        print(f"AI screening top {len(top_phrases)} {sentiment} phrases with {self.num_workers} workers...")
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i in range(0, len(top_phrases), batch_size):
                batch = top_phrases[i:i + batch_size]
                futures.append(executor.submit(process_screening_batch, batch))
            
            with tqdm(total=len(futures), desc="AI screening batches") as pbar:
                for future in as_completed(futures):
                    try:
                        batch_result = future.result()
                        screened_phrases.update(batch_result)
                    except Exception as e:
                        logger.error(f"Error in batch processing: {e}")
                    pbar.update(1)
                    time.sleep(0.1)  # Small delay between batches
        
        return screened_phrases
    
    def ai_merging_parallel(self, phrases: Dict[str, int], sentiment: str) -> Dict[str, int]:
        """Use AI to merge phrases with parallel processing."""
        if len(phrases) <= 1:
            return phrases
        
        # Sort phrases by frequency
        sorted_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
        
        # Group phrases into batches
        batch_size = 25
        merged_phrases = {}
        
        def process_merging_batch(batch):
            batch_merged = {}
            
            if len(batch) == 1:
                phrase, freq = batch[0]
                if phrase.strip():
                    batch_merged[phrase] = freq
                return batch_merged
            
            # Create prompt for AI merging
            phrase_list = [f"'{phrase}' (frequency: {freq})" for phrase, freq in batch]
            phrases_text = "\n".join(phrase_list)
            
            prompt = f"""Analyze these {sentiment} sentiment phrases about banking apps and aggressively merge those with similar meanings. 
            Combine frequencies for merged phrases. Choose the most appropriate phrase to represent each group.
            Be aggressive in merging - combine phrases that are similar in meaning.
            
            Phrases:
            {phrases_text}
            
            Return the result as a JSON array of objects with 'phrase' and 'frequency' fields.
            Merge phrases that are similar in meaning, even if not identical.
            
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
        
        print(f"AI merging {len(phrases)} {sentiment} phrases with {self.num_workers} workers...")
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i in range(0, len(sorted_phrases), batch_size):
                batch = sorted_phrases[i:i + batch_size]
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
                    time.sleep(0.1)  # Small delay between batches
        
        return merged_phrases
    
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
    
    def save_processed_phrases_to_txt(self, phrases: Dict[str, int], filename: str):
        """Save processed phrases to a text file."""
        os.makedirs('processed_phrases', exist_ok=True)
        filepath = os.path.join('processed_phrases', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"All Banks High Performance - Processed Phrases\n")
            f.write("="*50 + "\n\n")
            
            # Sort by frequency (descending)
            sorted_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
            
            for phrase, freq in sorted_phrases:
                f.write(f"'{phrase}' (frequency: {freq})\n")
        
        logger.info(f"Saved processed phrases to {filepath}")
    
    def process_bank_file(self, bank: str, sentiment: str) -> Dict[str, int]:
        """Process a single bank file with high performance AI enhancement."""
        filename = f"{bank}_{sentiment}_phrases.txt"
        filepath = os.path.join('.', filename)
        
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return {}
        
        print(f"\nProcessing {filename} with {self.num_workers} workers...")
        
        # Parse phrases
        phrases = self.parse_phrase_file(filepath)
        
        if not phrases:
            return {}
        
        # Round 1: Translate non-English phrases
        print("Round 1: Translating phrases...")
        translated_phrases = self.translate_phrases_parallel(phrases)
        
        # Round 2: Aggressive filtering with unclear phrase removal
        print("Round 2: Aggressive filtering with unclear phrase removal...")
        filtered_phrases = self.aggressive_filtering_with_unclear_removal(translated_phrases)
        
        # Round 3: AI screening with parallel processing
        print("Round 3: AI screening with parallel processing...")
        screened_phrases = self.ai_screening_parallel(filtered_phrases, sentiment)
        
        # Round 4: AI merging with parallel processing
        print("Round 4: AI merging with parallel processing...")
        final_phrases = self.ai_merging_parallel(screened_phrases, sentiment)
        
        # Round 5: Final merging if still too many phrases
        if len(final_phrases) > 100:
            print("Round 5: Final AI merging...")
            final_phrases = self.ai_merging_parallel(final_phrases, sentiment)
        
        # Save processed phrases to text file
        processed_filename = f"{bank}_{sentiment}_all_banks_high_performance_processed_phrases.txt"
        self.save_processed_phrases_to_txt(final_phrases, processed_filename)
        
        return final_phrases
    
    def process_all_banks(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Process all banks with high performance AI enhancement."""
        banks = ['welab_bank', 'za_bank']
        sentiments = ['positive', 'negative']
        
        results = {}
        
        print(f"Processing all banks with {self.num_workers} workers...")
        
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
                
                # Small delay between processing different files
                time.sleep(2)
        
        return results
    
    def generate_wordcloud(self, phrases: Dict[str, int], sentiment: str, bank: str, filename: str):
        """Generate a word cloud from phrases with custom colors."""
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
        
        # Adjust word cloud parameters based on phrase count
        max_words = min(200, len(phrases))  # More words for better visibility
        min_font_size = 8 if len(phrases) > 100 else 12
        max_font_size = 100 if len(phrases) > 100 else 120
        
        wordcloud = WordCloud(
            width=1200,  # Larger square format
            height=1200,  # Larger square format
            background_color='white',
            max_words=max_words,
            relative_scaling=0.7,  # Better word size distribution
            collocations=False,  # Avoid repeated words
            prefer_horizontal=0.6,  # Mix horizontal and vertical text
            min_font_size=min_font_size,
            max_font_size=max_font_size,
            color_func=color_func
        ).generate_from_frequencies(phrases)
        
        plt.figure(figsize=(14, 14))  # Larger square figure
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')  # Higher DPI for better quality
        plt.close()
        
        logger.info(f"Generated wordcloud: {filename}")
    
    def generate_all_wordclouds(self, results: Dict[str, Dict[str, Dict[str, int]]]):
        """Generate word clouds for all banks and sentiments."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        print("\nGenerating all banks high performance word clouds...")
        
        for bank, sentiments in results.items():
            for sentiment, phrases in sentiments.items():
                if phrases:
                    filename = f"{OUTPUT_DIR}/{bank}_{sentiment}_all_banks_high_performance_wordcloud.png"
                    self.generate_wordcloud(phrases, sentiment, bank, filename)
    
    def save_processed_results(self, results: Dict[str, Dict[str, Dict[str, int]]], filename: str = "all_banks_high_performance_processed_phrases.json"):
        """Save processed results to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved all banks high performance processed results to {filename}")
    
    def print_summary(self, results: Dict[str, Dict[str, Dict[str, int]]]):
        """Print a summary of the processing results."""
        print("\n" + "="*60)
        print("ALL BANKS HIGH PERFORMANCE PROCESSING SUMMARY")
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
    """Main function to run the all banks high performance processor."""
    api_key = DEEPSEEK_API_KEY
    
    if not api_key:
        api_key = input("Please enter your DeepSeek API key: ").strip()
        
    if not api_key:
        print("Error: API key is required")
        print("You can either:")
        print("1. Add your API key to the DEEPSEEK_API_KEY variable in config.py")
        print("2. Enter it when prompted")
        return
    
    print("All Banks High Performance AI Processor")
    print("="*50)
    print(f"Positive color: RGB{AllBanksHighPerformanceProcessor(api_key).positive_color}")
    print(f"Negative color: RGB{AllBanksHighPerformanceProcessor(api_key).negative_color}")
    print(f"Backing up existing wordclouds...")
    
    processor = AllBanksHighPerformanceProcessor(api_key, num_workers=15)
    
    # Backup existing wordclouds
    backup_dir = processor.backup_existing_wordclouds()
    
    print("Processing all banks with high performance AI enhancement...")
    start_time = time.time()
    
    results = processor.process_all_banks()
    
    processor.save_processed_results(results)
    
    print("Generating all banks high performance word clouds...")
    processor.generate_all_wordclouds(results)
    
    processor.print_summary(results)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\nAll banks high performance AI processing complete!")
    print(f"Total processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)")
    print(f"Backup location: {backup_dir}")
    print(f"Processed phrases saved to 'processed_phrases/' directory")
    print(f"Check the '{OUTPUT_DIR}' directory for the generated wordclouds.")

if __name__ == "__main__":
    main() 