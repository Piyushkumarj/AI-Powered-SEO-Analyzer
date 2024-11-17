import requests
from bs4 import BeautifulSoup
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import nltk
import re
import warnings
from urllib.parse import urljoin
warnings.filterwarnings('ignore')

class SEOAnalyzer:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('taggers/averaged_perceptron_tagger')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        
    def fetch_content(self, url):
        """Fetch website content and return BeautifulSoup object"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            print(f"Error fetching content: {str(e)}")
            return None

    def extract_metadata(self, soup):
        """Extract meta tags and important SEO elements"""
        metadata = {
            'title': soup.title.string if soup.title else None,
            'meta_description': None,
            'meta_keywords': None,
            'h1_tags': [],
            'h2_tags': [],
            'img_alt_texts': []
        }
        
        # Extract meta tags
        for meta in soup.find_all('meta'):
            if meta.get('name', '').lower() == 'description':
                metadata['meta_description'] = meta.get('content', '')
            elif meta.get('name', '').lower() == 'keywords':
                metadata['meta_keywords'] = meta.get('content', '')
        
        # Extract headers
        metadata['h1_tags'] = [h1.get_text() for h1 in soup.find_all('h1')]
        metadata['h2_tags'] = [h2.get_text() for h2 in soup.find_all('h2')]
        
        # Extract image alt texts
        metadata['img_alt_texts'] = [img.get('alt', '') for img in soup.find_all('img') if img.get('alt')]
        
        return metadata

    def extract_text_content(self, soup):
        """Extract main text content from the webpage"""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        return text

    def analyze_keywords(self, text, num_keywords=10):
        """Extract important keywords using TF-IDF"""
        # Tokenize and clean text
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
        
        # Get parts of speech and keep only nouns and adjectives
        pos_tags = pos_tag(tokens)
        filtered_tokens = [word for word, pos in pos_tags if pos.startswith(('NN', 'JJ'))]
        
        # Create document for TF-IDF
        document = [' '.join(filtered_tokens)]
        
        # Calculate TF-IDF
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=num_keywords)
        tfidf_matrix = vectorizer.fit_transform(document)
        
        # Get feature names and scores
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        # Create keyword-score pairs and sort
        keyword_scores = [(feature_names[i], scores[i]) for i in range(len(feature_names))]
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        return keyword_scores

    def analyze_seo(self, url):
        """Main method to analyze website SEO"""
        soup = self.fetch_content(url)
        if not soup:
            return None
        
        # Get metadata and content
        metadata = self.extract_metadata(soup)
        content = self.extract_text_content(soup)
        
        # Analyze keywords
        keywords = self.analyze_keywords(content)
        
        # Analyze content structure
        sentences = sent_tokenize(content)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(metadata, keywords, sentences)
        
        return {
            'metadata': metadata,
            'top_keywords': keywords,
            'recommendations': recommendations,
            'content_stats': {
                'word_count': len(content.split()),
                'sentence_count': len(sentences),
                'avg_sentence_length': np.mean([len(s.split()) for s in sentences])
            }
        }

    def generate_recommendations(self, metadata, keywords, sentences):
        """Generate SEO recommendations based on analysis"""
        recommendations = []
        
        # Title recommendations
        if not metadata['title']:
            recommendations.append("Missing page title - add a descriptive title tag")
        elif len(metadata['title']) < 30 or len(metadata['title']) > 60:
            recommendations.append("Title length should be between 30-60 characters")
            
        # Meta description recommendations
        if not metadata['meta_description']:
            recommendations.append("Missing meta description - add a compelling description")
        elif len(metadata['meta_description']) < 120 or len(metadata['meta_description']) > 160:
            recommendations.append("Meta description should be between 120-160 characters")
            
        # Header recommendations
        if not metadata['h1_tags']:
            recommendations.append("Missing H1 tag - add a primary heading")
        elif len(metadata['h1_tags']) > 1:
            recommendations.append("Multiple H1 tags found - consider using only one main heading")
            
        # Content recommendations
        avg_sentence_length = np.mean([len(s.split()) for s in sentences])
        if avg_sentence_length > 20:
            recommendations.append("Average sentence length is too high - consider making content more concise")
            
        # Keyword recommendations
        if keywords:
            top_keyword = keywords[0][0]
            if metadata['title'] and top_keyword not in metadata['title'].lower():
                recommendations.append(f"Consider including main keyword '{top_keyword}' in the title")
            if metadata['meta_description'] and top_keyword not in metadata['meta_description'].lower():
                recommendations.append(f"Consider including main keyword '{top_keyword}' in the meta description")
                
        return recommendations

def main():
    # Example usage
    analyzer = SEOAnalyzer()
    url = input("Enter website URL to analyze: ")
    results = analyzer.analyze_seo(url)
    
    if results:
        print("\n=== SEO Analysis Results ===")
        
        print("\nMetadata:")
        print(f"Title: {results['metadata']['title']}")
        print(f"Meta Description: {results['metadata']['meta_description']}")
        
        print("\nTop Keywords (with TF-IDF scores):")
        for keyword, score in results['top_keywords'][:10]:
            print(f"- {keyword}: {score:.4f}")
        
        print("\nContent Statistics:")
        print(f"Word Count: {results['content_stats']['word_count']}")
        print(f"Average Sentence Length: {results['content_stats']['avg_sentence_length']:.1f} words")
        
        print("\nSEO Recommendations:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i}. {rec}")

if __name__ == "__main__":
    main()