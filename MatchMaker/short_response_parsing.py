# UW Penpals Matchmaking Script
# 1/30/2021
# Author Kristofer Wong
# NLP code credit: Rupert Thomas
#   -> https://towardsdatascience.com/how-to-rank-text-content-by-semantic-similarity-4d2419a84c32

# IMPORTS:
import numpy as np

# For TFIDF
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# For fuzzy comparisons
from fuzzywuzzy import fuzz

# For Semantic comparisons
from re import sub
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import SoftCosineSimilarity
import gensim.downloader as api


# TFIDF similarity calculations
def calculate_tfidf_similarity(query, documents):
    # For use for calculating similarities in major/minor and hobbies

    stop_words = set(nltk.corpus.stopwords.words('english'))

    # Interface lemma tokenizer from nltk with sklearn
    class LemmaTokenizer:
        ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']

        def __init__(self):
            self.wnl = nltk.stem.WordNetLemmatizer()

        def __call__(self, doc):
            return [self.wnl.lemmatize(t) for t in nltk.word_tokenize(doc) if t not in self.ignore_tokens]

    # Lemmatize the stop words
    tokenizer = LemmaTokenizer()
    token_stop = tokenizer(' '.join(stop_words))

    # Create TF-idf model
    vectorizer = TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer)
    doc_vectors = vectorizer.fit_transform([str(query)] + list(documents))

    # Calculate similarity
    cosine_similarities = linear_kernel(doc_vectors[0:1], doc_vectors).flatten()

    return [item.item() for item in cosine_similarities[1:]]


# Fuzzy similarity calculations
def calculate_fuzzy_similarity(query, documents):
    sim = []
    for doc in documents:
        sim.append(fuzz.token_set_ratio(query, doc))
    return sim


# FUNCTIONS TO COMPARE SEMANTIC SIMILARITY OF TEXT USING COSINE SIMILARITY
def preprocess(doc, stopwords):
    # Tokenize, clean up input document string
    doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
    doc = sub(r'<[^<>]+(>|$)', " ", doc)
    doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
    doc = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
    return [token for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if token not in stopwords]


def preprocess_semantic_similarity(documents, glove):
    if documents.tolist().count('') != len(documents):
        return

    # For use for calculating similarities in about and want categories
    stopwords = list(nltk.corpus.stopwords.words('english'))

    # Preprocess the documents, including the query string
    corpus = [preprocess(document, stopwords) for document in documents]

    # Load the model: this is a big file, can take a while to download and open
    similarity_index = WordEmbeddingSimilarityIndex(glove)

    # Build the term dictionary, TF-idf model
    dictionary = Dictionary(corpus)
    tfidf = TfidfModel(dictionary=dictionary)

    # Create the term similarity matrix.
    similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf)

    return [dictionary, tfidf, similarity_matrix, corpus, stopwords]


def calculate_semantic_similarity(query, preprocessed_array):
    dictionary = preprocessed_array[0]
    tfidf = preprocessed_array[1]
    similarity_matrix = preprocessed_array[2]
    corpus = preprocessed_array[3]
    stopwords = preprocessed_array[4]

    query = preprocess(query, stopwords)
    query_tf = tfidf[dictionary.doc2bow(query)]

    index = SoftCosineSimilarity(
        tfidf[[dictionary.doc2bow(document) for document in corpus]],
        similarity_matrix)

    doc_similarity_scores = index[query_tf]

    return list(doc_similarity_scores)
