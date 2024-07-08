import re
import string

import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer

stemmer = SnowballStemmer('english')


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def topic_model_preprocess(text):
    """
    - Make text lowercase, remove text in square brackets,
    remove punctuation and remove words containing numbers.
    - Using gensim simple_preprocess
    - Remove STOPWORDS
    - Remove all words having length less than 3
    - Lemmatize & Stem the words
    """
    text = str(text)
    result = []
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)

    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result
