#!/usr/bin/env python3
"""
Enhanced Sentiment Phrase Processor
Addresses issues with generic words, personal pronouns, neutral words, and mixed sentiment.
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

class EnhancedSentimentProcessor:
    def __init__(self, deepseek_api_key: str):
        self.deepseek_api_key = deepseek_api_key
        self.api_base_url = DEEPSEEK_API_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
        # Define stop words and filters
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
            'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
            'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'having', 'do', 'does', 'did', 'doing', 'would', 'could', 'should', 'ought', 'im', 'youre', 'hes',
            'shes', 'its', 'were', 'theyre', 'ive', 'youve', 'weve', 'theyve', 'id', 'youd', 'hed', 'shed',
            'wed', 'theyd', 'ill', 'youll', 'hell', 'shell', 'well', 'theyll', 'isnt', 'arent', 'wasnt',
            'werent', 'hasnt', 'havent', 'hadnt', 'doesnt', 'dont', 'didnt', 'wont', 'wouldnt', 'couldnt',
            'shouldnt', 'lets', 'thats', 'whos', 'whats', 'heres', 'theres', 'whens', 'wheres', 'whys',
            'hows', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'don',
            'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn',
            'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn',
            'weren', 'won', 'wouldn'
        }
        
        # Neutral words that should be filtered out
        self.neutral_words = {
            'customer service', 'service', 'app', 'bank', 'banking', 'account', 'money', 'payment',
            'transfer', 'transaction', 'login', 'password', 'verification', 'security', 'update',
            'version', 'system', 'process', 'procedure', 'function', 'feature', 'interface', 'ui',
            'ux', 'design', 'layout', 'screen', 'page', 'button', 'menu', 'option', 'setting',
            'configuration', 'data', 'information', 'file', 'document', 'record', 'history',
            'log', 'report', 'status', 'result', 'response', 'message', 'notification', 'alert',
            'error', 'warning', 'success', 'failure', 'problem', 'issue', 'bug', 'glitch', 'crash',
            'freeze', 'hang', 'slow', 'fast', 'speed', 'performance', 'efficiency', 'quality',
            'reliability', 'stability', 'compatibility', 'accessibility', 'usability', 'functionality'
        }
        
        # Positive words that shouldn't appear in negative phrases
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'perfect', 'awesome',
            'brilliant', 'outstanding', 'superb', 'terrific', 'fabulous', 'marvelous', 'splendid',
            'magnificent', 'exceptional', 'extraordinary', 'incredible', 'unbelievable', 'remarkable',
            'impressive', 'satisfying', 'pleasing', 'enjoyable', 'delightful', 'lovely', 'beautiful',
            'nice', 'pleasant', 'comfortable', 'convenient', 'easy', 'simple', 'smooth', 'fast',
            'quick', 'efficient', 'effective', 'reliable', 'stable', 'secure', 'safe', 'trustworthy',
            'helpful', 'useful', 'valuable', 'beneficial', 'advantageous', 'profitable', 'successful',
            'working', 'functioning', 'operational', 'available', 'accessible', 'user-friendly',
            'intuitive', 'straightforward', 'clear', 'understandable', 'transparent', 'honest',
            'fair', 'reasonable', 'affordable', 'cheap', 'inexpensive', 'economical', 'cost-effective'
        }
        
        # Negative words that shouldn't appear in positive phrases
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'dreadful', 'atrocious', 'abysmal', 'appalling',
            'disgusting', 'revolting', 'nauseating', 'sickening', 'vile', 'foul', 'rotten', 'corrupt',
            'broken', 'damaged', 'defective', 'faulty', 'malfunctioning', 'non-working', 'useless',
            'worthless', 'pointless', 'meaningless', 'unnecessary', 'redundant', 'repetitive',
            'boring', 'tedious', 'monotonous', 'dull', 'uninteresting', 'unappealing', 'unattractive',
            'ugly', 'hideous', 'repulsive', 'offensive', 'insulting', 'rude', 'impolite', 'disrespectful',
            'unprofessional', 'incompetent', 'inefficient', 'ineffective', 'unreliable', 'unstable',
            'insecure', 'unsafe', 'dangerous', 'risky', 'hazardous', 'harmful', 'damaging', 'destructive',
            'expensive', 'costly', 'overpriced', 'unaffordable', 'unreasonable', 'unfair', 'dishonest',
            'deceptive', 'misleading', 'confusing', 'complicated', 'complex', 'difficult', 'hard',
            'challenging', 'frustrating', 'annoying', 'irritating', 'bothersome', 'troublesome',
            'problematic', 'troublesome', 'worrisome', 'concerning', 'alarming', 'disturbing',
            'upsetting', 'disappointing', 'dissatisfying', 'unsatisfactory', 'inadequate', 'insufficient',
            'incomplete', 'partial', 'limited', 'restricted', 'blocked', 'prevented', 'stopped',
            'failed', 'crashed', 'froze', 'hung', 'stuck', 'trapped', 'lost', 'missing', 'gone',
            'disappeared', 'vanished', 'erased', 'deleted', 'removed', 'eliminated', 'destroyed',
            'ruined', 'wasted', 'squandered', 'thrown away', 'discarded', 'abandoned', 'neglected',
            'ignored', 'overlooked', 'forgotten', 'unknown', 'unclear', 'uncertain', 'doubtful',
            'suspicious', 'questionable', 'untrustworthy', 'unreliable', 'unstable', 'inconsistent',
            'unpredictable', 'uncontrollable', 'unmanageable', 'unusable', 'inaccessible', 'unavailable'
        }
    
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
            
            response = requests.post(self.api_base_url, headers=self.headers, json=payload, timeout=API_TIMEOUT)
            response.raise_for_status()
            
            result = response.json()
            translated = result['choices'][0]['message']['content'].strip()
            translated = re.sub(r'^["\']|["\']$', '', translated)
            
            logger.info(f"Translated: '{text}' -> '{translated}'")
            return translated
            
        except Exception as e:
            logger.error(f"Translation failed for '{text}': {e}")
            return text
    
    def classify_sentiment_with_ai(self, phrase: str, expected_sentiment: str) -> Tuple[str, float]:
        """Use DeepSeek to classify sentiment and get confidence score."""
        try:
            prompt = f"""Analyze the sentiment of this phrase and determine if it matches the expected sentiment category.

Phrase: "{phrase}"
Expected sentiment: {expected_sentiment}

Please respond with:
1. The actual sentiment (positive/negative/neutral)
2. A confidence score from 0.0 to 1.0
3. Whether this phrase should be included in {expected_sentiment} sentiment analysis

Format your response as JSON:
{{
    "sentiment": "positive/negative/neutral",
    "confidence": 0.85,
    "include_in_{expected_sentiment}": true/false,
    "reason": "brief explanation"
}}"""

            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 200,
                "temperature": 0.1
            }
            
            response = requests.post(self.api_base_url, headers=self.headers, json=payload, timeout=API_TIMEOUT)
            response.raise_for_status()
            
            result = response.json()
            response_text = result['choices'][0]['message']['content'].strip()
            
            # Try to parse JSON response
            try:
                analysis = json.loads(response_text)
                return analysis.get('sentiment', 'neutral'), analysis.get('confidence', 0.5), analysis.get(f'include_in_{expected_sentiment}', True)
            except json.JSONDecodeError:
                # Fallback parsing
                if 'positive' in response_text.lower():
                    return 'positive', 0.7, True
                elif 'negative' in response_text.lower():
                    return 'negative', 0.7, True
                else:
                    return 'neutral', 0.5, False
                    
        except Exception as e:
            logger.error(f"AI sentiment classification failed for '{phrase}': {e}")
            return 'neutral', 0.5, True  # Default to include if AI fails
    
    def filter_phrase(self, phrase: str, sentiment: str) -> bool:
        """Filter out phrases that don't meet quality criteria."""
        phrase_lower = phrase.lower()
        words = phrase_lower.split()
        
        # 1. Filter out stop words and generic terms
        if len(words) <= 2:
            return False
        
        # Check if phrase is mostly stop words
        stop_word_count = sum(1 for word in words if word in self.stop_words)
        if stop_word_count / len(words) > 0.7:  # More than 70% stop words
            return False
        
        # 2. Filter out personal pronouns
        personal_pronouns = {'i', 'me', 'my', 'myself', 'we', 'us', 'our', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves'}
        if any(pronoun in words for pronoun in personal_pronouns):
            return False
        
        # 3. Filter out neutral words
        if any(neutral in phrase_lower for neutral in self.neutral_words):
            return False
        
        # 4. Check for sentiment consistency
        if sentiment == 'positive':
            # Check for negative words in positive phrases
            if any(negative in phrase_lower for negative in self.negative_words):
                return False
        elif sentiment == 'negative':
            # Check for positive words in negative phrases
            if any(positive in phrase_lower for positive in self.positive_words):
                return False
        
        # 5. Filter out very short or very long phrases
        if len(phrase) < 3 or len(phrase) > 100:
            return False
        
        # 6. Filter out phrases with too many numbers or special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z\s]', phrase)) / len(phrase)
        if special_char_ratio > 0.3:  # More than 30% special characters
            return False
        
        return True
    
    def similarity_score(self, phrase1: str, phrase2: str) -> float:
        """Calculate similarity between two phrases."""
        return SequenceMatcher(None, phrase1.lower(), phrase2.lower()).ratio()
    
    def merge_similar_phrases(self, phrases: Dict[str, int], similarity_threshold: float = SIMILARITY_THRESHOLD) -> Dict[str, int]:
        """Merge phrases with similar meanings and add up their frequencies."""
        if not phrases:
            return phrases
            
        sorted_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
        
        merged = {}
        used_indices = set()
        
        for i, (phrase1, freq1) in enumerate(sorted_phrases):
            if i in used_indices:
                continue
                
            total_freq = freq1
            best_phrase = phrase1
            
            for j, (phrase2, freq2) in enumerate(sorted_phrases[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                if self.similarity_score(phrase1, phrase2) >= similarity_threshold:
                    total_freq += freq2
                    used_indices.add(j)
                    
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
                
            pattern = r"'([^']+)' \(frequency: (\d+)\)"
            matches = re.findall(pattern, content)
            
            for phrase, freq in matches:
                phrases[phrase] = int(freq)
                
            logger.info(f"Parsed {len(phrases)} phrases from {filepath}")
            return phrases
            
        except Exception as e:
            logger.error(f"Error parsing {filepath}: {e}")
            return {}
    
    def process_phrases_with_ai_filtering(self, phrases: Dict[str, int], sentiment: str) -> Dict[str, int]:
        """Process phrases with AI-powered filtering and sentiment classification."""
        filtered_phrases = {}
        
        for phrase, freq in phrases.items():
            # Step 1: Basic filtering
            if not self.filter_phrase(phrase, sentiment):
                continue
            
            # Step 2: AI sentiment classification (for a sample to avoid too many API calls)
            if freq >= 3:  # Only check high-frequency phrases with AI
                ai_sentiment, confidence, should_include = self.classify_sentiment_with_ai(phrase, sentiment)
                
                if not should_include or confidence < 0.6:
                    continue
                
                logger.info(f"AI classified '{phrase}' as {ai_sentiment} (confidence: {confidence})")
            
            # Step 3: Add to filtered phrases
            filtered_phrases[phrase] = freq
            
            # Add delay to avoid API rate limits
            time.sleep(API_RETRY_DELAY)
        
        logger.info(f"Filtered {len(phrases)} phrases to {len(filtered_phrases)} phrases for {sentiment}")
        return filtered_phrases
    
    def process_all_files(self, similarity_threshold: float = SIMILARITY_THRESHOLD) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Process all sentiment phrase files with enhanced filtering."""
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
                
                # Apply AI-powered filtering
                filtered_phrases = self.process_phrases_with_ai_filtering(translated_phrases, sentiment)
                
                # Merge similar phrases
                merged_phrases = self.merge_similar_phrases(filtered_phrases, similarity_threshold)
                
                results[bank][sentiment] = merged_phrases
                
                time.sleep(API_RETRY_DELAY)
        
        return results
    
    def generate_wordcloud(self, phrases: Dict[str, int], title: str, filename: str):
        """Generate a word cloud from phrases."""
        if not phrases:
            logger.warning(f"No phrases to generate wordcloud for {title}")
            return
        
        wordcloud = WordCloud(
            width=WORDCLOUD_WIDTH,
            height=WORDCLOUD_HEIGHT,
            background_color='white',
            max_words=WORDCLOUD_MAX_WORDS,
            colormap=WORDCLOUD_COLORMAP,
            relative_scaling=WORDCLOUD_RELATIVE_SCALING
        ).generate_from_frequencies(phrases)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=20, pad=20)
        plt.tight_layout()
        
        plt.savefig(filename, dpi=WORDCLOUD_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated wordcloud: {filename}")
    
    def generate_all_wordclouds(self, results: Dict[str, Dict[str, Dict[str, int]]]):
        """Generate word clouds for all banks and sentiments."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        for bank, sentiments in results.items():
            for sentiment, phrases in sentiments.items():
                if phrases:
                    title = f"{bank.replace('_', ' ').title()} - {sentiment.title()} Phrases (Enhanced)"
                    filename = f"{OUTPUT_DIR}/{bank}_{sentiment}_enhanced_wordcloud.png"
                    self.generate_wordcloud(phrases, title, filename)
    
    def save_processed_results(self, results: Dict[str, Dict[str, Dict[str, int]]], filename: str = "enhanced_processed_phrases.json"):
        """Save processed results to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved enhanced processed results to {filename}")
    
    def print_summary(self, results: Dict[str, Dict[str, Dict[str, int]]]):
        """Print a summary of the processing results."""
        print("\n" + "="*60)
        print("ENHANCED PROCESSING SUMMARY")
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
    """Main function to run the enhanced sentiment processor."""
    api_key = DEEPSEEK_API_KEY
    
    if not api_key:
        api_key = input("Please enter your DeepSeek API key: ").strip()
        
    if not api_key:
        print("Error: API key is required")
        print("You can either:")
        print("1. Add your API key to the DEEPSEEK_API_KEY variable in config.py")
        print("2. Enter it when prompted")
        return
    
    processor = EnhancedSentimentProcessor(api_key)
    
    print("Processing sentiment phrases with enhanced filtering...")
    results = processor.process_all_files(similarity_threshold=SIMILARITY_THRESHOLD)
    
    processor.save_processed_results(results)
    
    print("Generating enhanced word clouds...")
    processor.generate_all_wordclouds(results)
    
    processor.print_summary(results)
    
    print("\nEnhanced processing complete! Check the 'wordclouds' directory for generated images.")

if __name__ == "__main__":
    main() 