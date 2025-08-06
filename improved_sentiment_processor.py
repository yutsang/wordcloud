#!/usr/bin/env python3
"""
Improved Sentiment Phrase Processor
Addresses issues with generic words, personal pronouns, neutral words, and mixed sentiment.
Uses efficient filtering without excessive API calls.
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

class ImprovedSentimentProcessor:
    def __init__(self, deepseek_api_key: str):
        self.deepseek_api_key = deepseek_api_key
        self.api_base_url = DEEPSEEK_API_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
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
            'werent', 'hasnt', 'havent', 'hadnt', 'doesnt', 'dont', 'didnt', 'wont', 'wouldnt',
            'couldnt', 'shouldnt', 'lets', 'thats', 'whos', 'whats', 'heres', 'theres', 'whens',
            'wheres', 'whys', 'hows', 'us', 'him', 'her', 'them', 'their', 'ours', 'yours',
            'mine', 'yours', 'his', 'hers', 'theirs', 'myself', 'yourself', 'himself', 'herself',
            'itself', 'ourselves', 'yourselves', 'themselves'
        }
        
        # Neutral/technical terms that don't convey sentiment
        self.neutral_terms = {
            'customer service', 'service', 'app', 'bank', 'banking', 'account', 'money', 'payment',
            'transfer', 'transaction', 'login', 'password', 'verification', 'security', 'update',
            'version', 'system', 'process', 'procedure', 'function', 'feature', 'interface', 'ui',
            'ux', 'design', 'layout', 'screen', 'page', 'button', 'menu', 'option', 'setting',
            'configuration', 'data', 'information', 'file', 'document', 'record', 'history',
            'log', 'report', 'status', 'result', 'response', 'message', 'notification', 'alert',
            'error', 'warning', 'success', 'failure', 'problem', 'issue', 'bug', 'glitch', 'crash',
            'freeze', 'hang', 'slow', 'fast', 'speed', 'performance', 'efficiency', 'quality',
            'reliability', 'stability', 'compatibility', 'accessibility', 'usability', 'functionality',
            'time', 'day', 'week', 'month', 'year', 'today', 'yesterday', 'tomorrow', 'now', 'then',
            'here', 'there', 'where', 'when', 'why', 'how', 'what', 'which', 'who', 'whom',
            'number', 'amount', 'quantity', 'size', 'length', 'width', 'height', 'weight', 'volume',
            'percent', 'percentage', 'rate', 'ratio', 'proportion', 'fraction', 'decimal', 'integer',
            'digit', 'figure', 'statistic', 'measurement', 'unit', 'scale', 'level', 'degree',
            'category', 'type', 'kind', 'sort', 'class', 'group', 'set', 'collection', 'series',
            'list', 'table', 'chart', 'graph', 'diagram', 'image', 'picture', 'photo', 'video',
            'audio', 'sound', 'voice', 'text', 'word', 'sentence', 'paragraph', 'section', 'chapter',
            'book', 'document', 'paper', 'article', 'report', 'summary', 'description', 'explanation',
            'instruction', 'manual', 'guide', 'tutorial', 'example', 'sample', 'test', 'trial',
            'experiment', 'study', 'research', 'analysis', 'evaluation', 'assessment', 'review',
            'comment', 'feedback', 'opinion', 'view', 'perspective', 'attitude', 'feeling', 'emotion',
            'mood', 'tone', 'style', 'manner', 'way', 'method', 'approach', 'strategy', 'plan',
            'scheme', 'program', 'project', 'task', 'job', 'work', 'activity', 'action', 'operation',
            'procedure', 'process', 'step', 'stage', 'phase', 'period', 'duration', 'interval',
            'moment', 'instant', 'second', 'minute', 'hour', 'morning', 'afternoon', 'evening', 'night'
        }
        
        # Positive sentiment words
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
            'fair', 'reasonable', 'affordable', 'cheap', 'inexpensive', 'economical', 'cost-effective',
            'love', 'like', 'enjoy', 'appreciate', 'admire', 'respect', 'trust', 'believe', 'hope',
            'wish', 'want', 'need', 'desire', 'prefer', 'choose', 'select', 'pick', 'decide',
            'agree', 'accept', 'approve', 'support', 'recommend', 'suggest', 'advise', 'encourage',
            'motivate', 'inspire', 'excite', 'thrill', 'amaze', 'surprise', 'impress', 'satisfy',
            'please', 'delight', 'entertain', 'amuse', 'cheer', 'comfort', 'relax', 'calm', 'soothe',
            'heal', 'cure', 'fix', 'repair', 'improve', 'enhance', 'upgrade', 'update', 'modernize',
            'innovate', 'create', 'build', 'develop', 'design', 'plan', 'organize', 'arrange',
            'prepare', 'ready', 'complete', 'finish', 'accomplish', 'achieve', 'succeed', 'win',
            'gain', 'earn', 'profit', 'benefit', 'advantage', 'opportunity', 'chance', 'possibility',
            'potential', 'future', 'progress', 'advance', 'grow', 'increase', 'expand', 'extend',
            'enlarge', 'bigger', 'larger', 'more', 'better', 'best', 'superior', 'premium', 'premium',
            'luxury', 'exclusive', 'special', 'unique', 'rare', 'unusual', 'different', 'new',
            'fresh', 'clean', 'pure', 'natural', 'organic', 'healthy', 'strong', 'powerful',
            'energetic', 'active', 'dynamic', 'vibrant', 'lively', 'bright', 'shiny', 'sparkling',
            'glowing', 'radiant', 'beautiful', 'attractive', 'appealing', 'charming', 'cute',
            'adorable', 'sweet', 'kind', 'gentle', 'soft', 'smooth', 'silky', 'velvety', 'warm',
            'cozy', 'comfortable', 'relaxing', 'peaceful', 'quiet', 'calm', 'serene', 'tranquil'
        }
        
        # Negative sentiment words
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
            'problematic', 'worrisome', 'concerning', 'alarming', 'disturbing', 'upsetting',
            'disappointing', 'dissatisfying', 'unsatisfactory', 'inadequate', 'insufficient',
            'incomplete', 'partial', 'limited', 'restricted', 'blocked', 'prevented', 'stopped',
            'failed', 'crashed', 'froze', 'hung', 'stuck', 'trapped', 'lost', 'missing', 'gone',
            'disappeared', 'vanished', 'erased', 'deleted', 'removed', 'eliminated', 'destroyed',
            'ruined', 'wasted', 'squandered', 'thrown away', 'discarded', 'abandoned', 'neglected',
            'ignored', 'overlooked', 'forgotten', 'unknown', 'unclear', 'uncertain', 'doubtful',
            'suspicious', 'questionable', 'untrustworthy', 'unreliable', 'unstable', 'inconsistent',
            'unpredictable', 'uncontrollable', 'unmanageable', 'unusable', 'inaccessible', 'unavailable',
            'hate', 'dislike', 'loathe', 'despise', 'abhor', 'detest', 'resent', 'envy', 'jealous',
            'angry', 'mad', 'furious', 'rage', 'fury', 'wrath', 'irritated', 'annoyed', 'bothered',
            'troubled', 'worried', 'anxious', 'nervous', 'scared', 'afraid', 'frightened', 'terrified',
            'panicked', 'stressed', 'tension', 'pressure', 'strain', 'burden', 'load', 'weight',
            'pain', 'hurt', 'suffering', 'agony', 'torment', 'torture', 'misery', 'sorrow', 'grief',
            'sadness', 'depression', 'despair', 'hopelessness', 'helplessness', 'powerlessness',
            'weakness', 'frailty', 'fragility', 'vulnerability', 'exposure', 'risk', 'danger',
            'threat', 'menace', 'peril', 'hazard', 'jeopardy', 'endangerment', 'compromise',
            'violation', 'breach', 'infringement', 'invasion', 'intrusion', 'interference',
            'disruption', 'disturbance', 'interruption', 'obstruction', 'blockage', 'barrier',
            'obstacle', 'hindrance', 'impediment', 'restriction', 'limitation', 'constraint',
            'restraint', 'control', 'domination', 'oppression', 'suppression', 'repression',
            'inhibition', 'prevention', 'prohibition', 'ban', 'forbidden', 'illegal', 'unlawful',
            'criminal', 'guilty', 'blame', 'fault', 'error', 'mistake', 'wrong', 'incorrect',
            'false', 'fake', 'phony', 'fraud', 'scam', 'cheat', 'deceive', 'lie', 'falsehood',
            'untruth', 'fiction', 'myth', 'legend', 'story', 'tale', 'narrative', 'account',
            'report', 'statement', 'declaration', 'announcement', 'notice', 'warning', 'alert',
            'caution', 'advice', 'suggestion', 'recommendation', 'proposal', 'offer', 'deal',
            'bargain', 'discount', 'sale', 'promotion', 'advertisement', 'commercial', 'marketing',
            'publicity', 'exposure', 'visibility', 'recognition', 'awareness', 'knowledge',
            'understanding', 'comprehension', 'grasp', 'hold', 'grip', 'clutch', 'seize', 'catch',
            'capture', 'trap', 'snare', 'net', 'web', 'tangle', 'knot', 'tie', 'bind', 'fasten',
            'secure', 'lock', 'close', 'shut', 'seal', 'block', 'stop', 'halt', 'pause', 'wait',
            'delay', 'postpone', 'defer', 'suspend', 'freeze', 'hold', 'keep', 'retain', 'maintain',
            'preserve', 'protect', 'guard', 'defend', 'shield', 'cover', 'hide', 'conceal', 'mask',
            'disguise', 'camouflage', 'stealth', 'secret', 'private', 'confidential', 'classified',
            'restricted', 'limited', 'exclusive', 'special', 'unique', 'rare', 'unusual', 'strange',
            'odd', 'weird', 'bizarre', 'peculiar', 'curious', 'interesting', 'fascinating',
            'intriguing', 'mysterious', 'enigmatic', 'puzzling', 'confusing', 'perplexing',
            'bewildering', 'baffling', 'stumping', 'challenging', 'difficult', 'hard', 'tough',
            'rough', 'harsh', 'severe', 'strict', 'rigid', 'inflexible', 'stubborn', 'obstinate',
            'determined', 'resolute', 'firm', 'strong', 'powerful', 'mighty', 'forceful', 'intense',
            'fierce', 'violent', 'aggressive', 'hostile', 'antagonistic', 'oppositional', 'contrary',
            'opposite', 'reverse', 'backward', 'backwards', 'retrograde', 'regressive', 'declining',
            'falling', 'dropping', 'decreasing', 'reducing', 'diminishing', 'shrinking', 'contracting',
            'narrowing', 'tightening', 'constricting', 'squeezing', 'pressing', 'pushing', 'forcing',
            'compelling', 'coercing', 'pressuring', 'urging', 'encouraging', 'motivating', 'inspiring',
            'stimulating', 'exciting', 'thrilling', 'amazing', 'astonishing', 'surprising', 'shocking',
            'stunning', 'dazzling', 'brilliant', 'bright', 'shining', 'glowing', 'radiant', 'luminous',
            'illuminated', 'enlightened', 'educated', 'informed', 'knowledgeable', 'wise', 'intelligent',
            'smart', 'clever', 'bright', 'sharp', 'quick', 'fast', 'rapid', 'swift', 'speedy',
            'hasty', 'hurried', 'rushed', 'urgent', 'immediate', 'instant', 'momentary', 'brief',
            'short', 'temporary', 'transient', 'fleeting', 'passing', 'fading', 'disappearing',
            'vanishing', 'evaporating', 'dissolving', 'melting', 'softening', 'weakening', 'failing',
            'falling', 'collapsing', 'crumbling', 'breaking', 'shattering', 'splitting', 'cracking',
            'fracturing', 'rupturing', 'bursting', 'exploding', 'blowing', 'blasting', 'destroying',
            'demolishing', 'wrecking', 'ruining', 'spoiling', 'damaging', 'harming', 'hurting',
            'injuring', 'wounding', 'cutting', 'slashing', 'stabbing', 'piercing', 'penetrating',
            'entering', 'invading', 'attacking', 'assaulting', 'striking', 'hitting', 'beating',
            'pounding', 'hammering', 'banging', 'knocking', 'tapping', 'touching', 'feeling',
            'sensing', 'perceiving', 'noticing', 'observing', 'watching', 'looking', 'seeing',
            'viewing', 'examining', 'inspecting', 'checking', 'testing', 'trying', 'attempting',
            'endeavoring', 'striving', 'struggling', 'fighting', 'battling', 'competing', 'contesting',
            'challenging', 'opposing', 'resisting', 'defying', 'rejecting', 'refusing', 'denying',
            'disagreeing', 'disputing', 'arguing', 'debating', 'discussing', 'talking', 'speaking',
            'saying', 'telling', 'mentioning', 'referring', 'citing', 'quoting', 'repeating',
            'echoing', 'mirroring', 'reflecting', 'showing', 'displaying', 'presenting', 'offering',
            'providing', 'supplying', 'giving', 'handing', 'passing', 'transferring', 'moving',
            'shifting', 'changing', 'altering', 'modifying', 'adjusting', 'adapting', 'fitting',
            'matching', 'corresponding', 'relating', 'connecting', 'linking', 'joining', 'uniting',
            'combining', 'merging', 'mixing', 'blending', 'integrating', 'incorporating', 'including',
            'adding', 'appending', 'attaching', 'affixing', 'fastening', 'securing', 'fixing',
            'repairing', 'mending', 'healing', 'curing', 'treating', 'nursing', 'caring', 'tending',
            'managing', 'handling', 'dealing', 'coping', 'surviving', 'living', 'existing', 'being',
            'becoming', 'growing', 'developing', 'evolving', 'progressing', 'advancing', 'improving',
            'enhancing', 'upgrading', 'updating', 'modernizing', 'innovating', 'creating', 'building',
            'constructing', 'making', 'forming', 'shaping', 'molding', 'sculpting', 'carving',
            'cutting', 'trimming', 'pruning', 'cultivating', 'nurturing', 'fostering', 'promoting',
            'encouraging', 'supporting', 'backing', 'endorsing', 'approving', 'accepting', 'agreeing',
            'consenting', 'allowing', 'permitting', 'authorizing', 'sanctioning', 'validating',
            'confirming', 'verifying', 'checking', 'testing', 'proving', 'demonstrating', 'showing',
            'exhibiting', 'displaying', 'presenting', 'revealing', 'exposing', 'uncovering', 'discovering',
            'finding', 'locating', 'identifying', 'recognizing', 'acknowledging', 'admitting',
            'confessing', 'declaring', 'announcing', 'proclaiming', 'stating', 'saying', 'telling',
            'informing', 'notifying', 'alerting', 'warning', 'cautioning', 'advising', 'counseling',
            'guiding', 'directing', 'leading', 'conducting', 'orchestrating', 'coordinating',
            'organizing', 'arranging', 'planning', 'preparing', 'readying', 'setting', 'establishing',
            'founding', 'creating', 'building', 'developing', 'growing', 'expanding', 'extending',
            'enlarging', 'increasing', 'multiplying', 'doubling', 'tripling', 'quadrupling',
            'maximizing', 'optimizing', 'perfecting', 'refining', 'polishing', 'smoothing',
            'softening', 'gentling', 'calming', 'soothing', 'comforting', 'reassuring', 'encouraging',
            'inspiring', 'motivating', 'energizing', 'stimulating', 'exciting', 'thrilling',
            'amazing', 'astonishing', 'surprising', 'shocking', 'stunning', 'dazzling', 'impressing',
            'influencing', 'affecting', 'impacting', 'touching', 'moving', 'stirring', 'arousing',
            'awakening', 'waking', 'rising', 'ascending', 'climbing', 'scaling', 'reaching',
            'achieving', 'attaining', 'gaining', 'obtaining', 'acquiring', 'getting', 'receiving',
            'accepting', 'welcoming', 'embracing', 'adopting', 'choosing', 'selecting', 'picking',
            'electing', 'voting', 'deciding', 'determining', 'resolving', 'settling', 'concluding',
            'finishing', 'completing', 'ending', 'terminating', 'stopping', 'halting', 'pausing',
            'waiting', 'delaying', 'postponing', 'deferring', 'suspending', 'freezing', 'holding',
            'keeping', 'retaining', 'maintaining', 'preserving', 'protecting', 'guarding', 'defending',
            'shielding', 'covering', 'hiding', 'concealing', 'masking', 'disguising', 'camouflaging',
            'protecting', 'safeguarding', 'securing', 'locking', 'closing', 'shutting', 'sealing',
            'blocking', 'preventing', 'stopping', 'halting', 'arresting', 'checking', 'curbing',
            'restraining', 'controlling', 'managing', 'handling', 'dealing', 'coping', 'surviving',
            'enduring', 'persisting', 'continuing', 'lasting', 'remaining', 'staying', 'lingering',
            'waiting', 'pausing', 'stopping', 'halting', 'freezing', 'holding', 'keeping',
            'retaining', 'maintaining', 'preserving', 'protecting', 'guarding', 'defending',
            'shielding', 'covering', 'hiding', 'concealing', 'masking', 'disguising', 'camouflaging'
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
    
    def filter_phrase(self, phrase: str, sentiment: str) -> bool:
        """Filter out phrases that don't meet quality criteria."""
        phrase_lower = phrase.lower()
        words = phrase_lower.split()
        
        # 1. Filter out very short phrases
        if len(words) <= 2:
            return False
        
        # 2. Filter out phrases that are mostly stop words
        stop_word_count = sum(1 for word in words if word in self.stop_words)
        if stop_word_count / len(words) > 0.6:  # More than 60% stop words
            return False
        
        # 3. Filter out personal pronouns
        personal_pronouns = {'i', 'me', 'my', 'myself', 'we', 'us', 'our', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves'}
        if any(pronoun in words for pronoun in personal_pronouns):
            return False
        
        # 4. Filter out neutral/technical terms
        if any(neutral in phrase_lower for neutral in self.neutral_terms):
            return False
        
        # 5. Check for sentiment consistency
        if sentiment == 'positive':
            # Check for negative words in positive phrases
            negative_count = sum(1 for word in words if word in self.negative_words)
            if negative_count > 0:
                return False
        elif sentiment == 'negative':
            # Check for positive words in negative phrases
            positive_count = sum(1 for word in words if word in self.positive_words)
            if positive_count > 0:
                return False
        
        # 6. Filter out very long phrases
        if len(phrase) > 80:
            return False
        
        # 7. Filter out phrases with too many special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z\s]', phrase)) / len(phrase)
        if special_char_ratio > 0.25:  # More than 25% special characters
            return False
        
        # 8. Filter out phrases that are too generic
        generic_phrases = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        if all(word in generic_phrases for word in words):
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
    
    def process_phrases_with_filtering(self, phrases: Dict[str, int], sentiment: str) -> Dict[str, int]:
        """Process phrases with enhanced filtering."""
        filtered_phrases = {}
        
        for phrase, freq in phrases.items():
            if self.filter_phrase(phrase, sentiment):
                filtered_phrases[phrase] = freq
        
        logger.info(f"Filtered {len(phrases)} phrases to {len(filtered_phrases)} phrases for {sentiment}")
        return filtered_phrases
    
    def process_all_files(self, similarity_threshold: float = SIMILARITY_THRESHOLD) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Process all sentiment phrase files with improved filtering."""
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
                
                # Apply enhanced filtering
                filtered_phrases = self.process_phrases_with_filtering(translated_phrases, sentiment)
                
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
                    title = f"{bank.replace('_', ' ').title()} - {sentiment.title()} Phrases (Improved)"
                    filename = f"{OUTPUT_DIR}/{bank}_{sentiment}_improved_wordcloud.png"
                    self.generate_wordcloud(phrases, title, filename)
    
    def save_processed_results(self, results: Dict[str, Dict[str, Dict[str, int]]], filename: str = "improved_processed_phrases.json"):
        """Save processed results to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved improved processed results to {filename}")
    
    def print_summary(self, results: Dict[str, Dict[str, Dict[str, int]]]):
        """Print a summary of the processing results."""
        print("\n" + "="*60)
        print("IMPROVED PROCESSING SUMMARY")
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
    """Main function to run the improved sentiment processor."""
    api_key = DEEPSEEK_API_KEY
    
    if not api_key:
        api_key = input("Please enter your DeepSeek API key: ").strip()
        
    if not api_key:
        print("Error: API key is required")
        print("You can either:")
        print("1. Add your API key to the DEEPSEEK_API_KEY variable in config.py")
        print("2. Enter it when prompted")
        return
    
    processor = ImprovedSentimentProcessor(api_key)
    
    print("Processing sentiment phrases with improved filtering...")
    results = processor.process_all_files(similarity_threshold=SIMILARITY_THRESHOLD)
    
    processor.save_processed_results(results)
    
    print("Generating improved word clouds...")
    processor.generate_all_wordclouds(results)
    
    processor.print_summary(results)
    
    print("\nImproved processing complete! Check the 'wordclouds' directory for generated images.")

if __name__ == "__main__":
    main() 