# All text preprocess code is here

import os
import json
import re

import spacy
import unicodedata
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from bs4 import BeautifulSoup
from textblob import TextBlob
from textblob import Word
from textblob.sentiments import NaiveBayesAnalyzer
from spacy.lang.en.stop_words import STOP_WORDS as sw
import nltk

fpath = os.path.join(os.path.dirname(__file__), 'data', 'contractions.json')
contractions = json.load(open(fpath))

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    '''print("Downloading language model Use this...")
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')'''
    

# List of NLTK packages and their corresponding resource paths for checking
nltk_packages = [
    ('punkt_tab', 'tokenizers/punkt_tab'),
    ('stopwords', 'corpora/stopwords'),
    ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
    ('tagsets_json', 'help/tagsets_json'),
    ('wordnet', 'corpora/wordnet'),
    ('words', 'corpora/words'),
    ('maxent_ne_chunker_tab', 'chunkers/maxent_ne_chunker_tab'),
]

def download_nltk_packages():
    """
    Downloads the required NLTK packages.
    """
    for package, _ in nltk_packages:
        nltk.download(package)

# Check if NLTK packages are present; if not, instruct the user to download them
missing_packages = []
for package, resource in nltk_packages:
    try:
        nltk.data.find(resource)
    except LookupError:
        missing_packages.append(package)

if missing_packages:
    print(f"Warning: The following NLTK packages are missing: {', '.join(missing_packages)}")
    print("Please run the 'download_nltk_packages()' function to download them.")

# General Feature Extraction

def word_count(x):
        return len(x.split())

def char_count(x):
        pattern = r'\s'
        return len(re.sub(pattern, '', x))

def avg_word_len(x):
        return char_count(x)/word_count(x)

def stop_words_count(x):
        temp = len([word for word in x.lower().split() if word in sw])
        return temp

def hashtags_count(x):
        return len(re.findall(r'#\w+', x))

def mentions_count(x):
        return len(re.findall(r'@\w+', x))

def numerics_count(x):
        return len(re.findall(r'\b\d+\b', x))

def upper_case_count(x):
        return len([word for word in x.split() if word.isupper()])

# Preprocessing and Cleaning
def to_lower_case(x):
        return x.lower()

def contraction_to_expansion(x):
        return " ".join([contractions.get(word.lower(), word) for word in x.split()])

def remove_emails(x):
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(pattern, '', x)

def count_emails(x):
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z.-]+\.[A-Z|a-z]{2,}\b'
        return len(re.findall(pattern, '', x))

def count_urls(x):
        pattern = r'http\S+|www\.\S+'
        return len(re.findall(pattern, x))

def remove_urls(x):
        pattern = r'http\S+|www\.\S+'
        return re.sub(pattern, '', x)

def count_rt(x):
        pattern = r'\bRT @\w+'
        return len(re.findall(pattern, x))

def remove_rt(x):
        pattern = r'\bRT @\w+'
        return re.sub(pattern, '', x)

def remove_html_tag(x):
        return BeautifulSoup(x, 'lxml').get_text()

def remove_accented_chars(x):
        return unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def remove_mentions(x):
        pattern = r'@\w+'
        return re.sub(pattern, "", x).strip()              

def remove_special_chars(x):
        pattern = r'[^\w\s]'
        return re.sub(pattern,'',x)

def remove_repeated_chars(x):
        pattern = r'(.)\1+'
        return re.sub(pattern, r'\1\1', x)

def remove_stop_words(x):
        return " ".join([word for word in x.split() if word not in sw])

def convert_to_base(x):
        doc = nlp(x)
        tokens = []
        for token in doc:
                if token.pos_ in ['NOUN', 'VERB']:
                        tokens.append(token.lemma_)
                else:
                        tokens.append(token.text)
        x = ' '.join(tokens)
        pattern = r'\s\.'
        x = re.sub(pattern, '.', x)
        return x

def lemmatize(x):
        doc = nlp(x)
        return ' '.join([token.lemma_ for token in doc])

def get_wordcloud(x):
    
    cloud = WordCloud(width=800, height=500).generate(x)
    plt.figure(figsize=(15, 7), dpi=100)
    plt.imshow(cloud.to_image(), interpolation='bilinear')
    plt.axis('off')
    plt.show()

def correct_spelling(x):
        words = []
        for word in x.split():
                w = Word(word)
                words.append(w.correct())

        return ' '.join(words)

def get_noun_phrase(x):
        blob = TextBlob(x)
        noun_phrase = blob.noun_phrases
        return noun_phrase

def n_grams(x, n=2):
        return list(TextBlob(x).ngrams(n))

def singularize_words(x):
        blob = TextBlob(x)
        return ' '.join([word.singularize() if tag in ['NNS'] else word for word, tag in blob.tags])

def pluralize_words(x):
        blob = TextBlob(x)
        return ' '.join([word.singularize() if tag in ['NN'] else word for word, tag in blob.tags])

def sentiment_analysis(x):
        return TextBlob(x, analyzer=NaiveBayesAnalyzer()).sentiment.classification