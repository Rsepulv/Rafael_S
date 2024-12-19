'''
Author: Rafael Sepulveda    

Description: This program is a web scraper that crawls a website defined in the base_url variable. It extracts data such as unique urls, 
image urls, phone numbers, and zip codes. The program also cleans the text from the website, identifies unique words, nouns, and verbs. The
extracted data is then printed into a txt report.  
'''

import requests
import os
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
import string


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')  
stopwords_set = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
zipPatt = re.compile(r'\d{5}(?:-\d{4})?')
phonePatt = re.compile(r'\(?\d{3}\)? -?\d{3}-? *-?\d{4}')
dimension_patt = re.compile(r'\d+px|\d+em|\d+pt|\d+rem')
hash_patt = re.compile(r'[a-f0-9]{32,64}')

base_url = 'https://casl.website/' #link to Cyberapolis training website  


def fetch_page(url):
    try:
        response = requests.get(url)
        response.raise_for_status() 
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_urls(soup):
    urls = set()
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('http') or href.startswith('https'):
            if 'casl.website' in href:  
                urls.add(href)
    return urls

def extract_image_urls(soup):
    image_urls = set()
    for img in soup.find_all('img', src=True):
        img_url = img['src']
        if img_url.startswith('/'):
            img_url = base_url + img_url.lstrip('/')
        image_urls.add(img_url)
    return image_urls

def extract_phone_numbers(text):
    return set(phonePatt.findall(text))

def extract_zip_codes(text):
    return set(zipPatt.findall(text))

def filter_non_content_words(tokens):
    filtered_tokens = []
    for word in tokens:
        if word.lower() in stopwords_set or \
           re.match(dimension_patt, word) or \
           re.match(hash_patt, word) or \
           word in string.punctuation or \
           word.isnumeric() or \
           word.startswith('http') or \
           word.startswith('//') or \
           '=' in word:
            continue
        filtered_tokens.append(word)
    return filtered_tokens

def extract_vocabulary(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = filter_non_content_words(tokens)
    return set(filtered_tokens)

def refine_verbs_and_nouns(text):
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    refined_verbs = set()
    refined_nouns = set()
    for word, tag in pos_tags:
        lemmatized_word = lemmatizer.lemmatize(word.lower())
        if tag.startswith('VB') and lemmatized_word not in stopwords_set:
            refined_verbs.add(lemmatized_word)
        elif tag.startswith('NN') and lemmatized_word not in stopwords_set:
            refined_nouns.add(lemmatized_word)

    refined_verbs = filter_non_content_words(list(refined_verbs))
    refined_nouns = filter_non_content_words(list(refined_nouns))
    verb_freq = Counter(refined_verbs)
    noun_freq = Counter(refined_nouns)
    
    return refined_verbs, refined_nouns, verb_freq, noun_freq

def scrape_site(url):
    visited_urls = set()
    all_image_urls = set()
    all_phone_numbers = set()
    all_zip_codes = set()
    all_text = []

    pages_to_visit = [url]
    while pages_to_visit:
        current_url = pages_to_visit.pop()
        if current_url in visited_urls:
            continue

        page_content = fetch_page(current_url)
        if page_content:
            soup = BeautifulSoup(page_content, 'html.parser')
            visited_urls.add(current_url)

            for script in soup(["script", "style"]):
                script.decompose()

            text_content = soup.get_text(separator=" ", strip=True)
            all_text.append(text_content)

            all_image_urls.update(extract_image_urls(soup))
            all_phone_numbers.update(extract_phone_numbers(page_content))
            all_zip_codes.update(extract_zip_codes(page_content))

            new_urls = extract_urls(soup)
            pages_to_visit.extend(new_urls - visited_urls)

    all_text_content = '\n'.join(all_text)

    vocabulary = extract_vocabulary(all_text_content)
    verbs, nouns, verb_freq, noun_freq = refine_verbs_and_nouns(all_text_content)

    return {
        'unique_urls': visited_urls,
        'image_urls': all_image_urls,
        'phone_numbers': all_phone_numbers,
        'zip_codes': all_zip_codes,
        'vocabulary': vocabulary,
        'verbs': verbs,
        'nouns': nouns,
        'verb_freq': verb_freq,
        'noun_freq': noun_freq
    }

def generate_report(results):
    print("Report:")
    print("\nUnique URLs:")
    for url in results['unique_urls']:
        print(url)
    
    print("\nImage URLs:")
    for img_url in results['image_urls']:
        print(img_url)
    
    print("\nPhone Numbers:")
    for phone in results['phone_numbers']:
        print(phone)

    print("\nZip Codes:")
    for zip_code in results['zip_codes']:
        print(zip_code)

    print("\nVocabulary (Unique words):")
    for word in results['vocabulary']:
        print(word)

    print("\nVerbs:")
    for verb in results['verbs']:
        print(verb)

    print("\nNouns:")
    for noun in results['nouns']:
        print(noun)

    with open('report.txt', 'w') as report_file:
        report_file.write("Report:\n")
        report_file.write("\nUnique URLs:\n")
        for url in results['unique_urls']:
            report_file.write(url + "\n")

        report_file.write("\nImage URLs:\n")
        for img_url in results['image_urls']:
            report_file.write(img_url + "\n")

        report_file.write("\nPhone Numbers:\n")
        for phone in results['phone_numbers']:
            report_file.write(phone + "\n")

        report_file.write("\nZip Codes:\n")
        for zip_code in results['zip_codes']:
            report_file.write(zip_code + "\n")

        report_file.write("\nVocabulary (Unique words):\n")
        for word in results['vocabulary']:
            report_file.write(word + "\n")

        report_file.write("\nVerbs:\n")
        for verb in results['verbs']:
            report_file.write(verb + "\n")

        report_file.write("\nNouns:\n")
        for noun in results['nouns']:
            report_file.write(noun + "\n")

if __name__ == '__main__':
    print("Starting the web scraping process...")
    results = scrape_site(base_url)
    generate_report(results)