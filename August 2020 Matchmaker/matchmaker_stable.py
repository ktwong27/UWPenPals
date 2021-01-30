# UW Penpals Matchmaking Script
# 9/16/2020
# Author Kristofer Wong
# NLP code credit: Rupert Thomas
#   -> https://towardsdatascience.com/how-to-rank-text-content-by-semantic-similarity-4d2419a84c32


import sys
import csv
from enum import IntEnum
import numpy as np

import nltk
from re import sub
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import gensim.downloader as api
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import SoftCosineSimilarity


class Person(IntEnum):
    NAME = 1
    ADDRESS = 2
    MAJOR = 3
    YEAR = 4
    GENDER = 5
    PRONOUNS = 6
    PAL_GENDER = 7
    HOBBIES = 8
    ABOUT = 9
    SOCIAL = 10
    MULTIPLE = 11
    HOUSE = 12
    ELEMENT = 13
    OTHER = 14
    WANT = 15


class Weights(IntEnum):
    NAME = 0
    ADDRESS = 0
    MAJOR = 1
    YEAR = 1
    GENDER = 0
    PRONOUNS = 0
    PAL_GENDER = 0
    HOBBIES = 3
    ABOUT = 2
    SOCIAL = 0
    MULTIPLE = 0
    HOUSE = 1
    ELEMENT = 1
    OTHER = 0
    WANT = 4


def remove_duplicate_entries(responses):
    # Get rid of duplicates
    names = set()
    addresses = set()

    # reversed so we get rid of first submission.
    # people tend to submit multiple times to change something
    for response in reversed(responses):
        if response[Person.NAME] in names \
                and response[Person.ADDRESS] in addresses:
            responses.remove(response)
        names.add(response[Person.NAME])
        addresses.add(response[Person.ADDRESS])


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


def preprocess(doc, stopwords):
    # Tokenize, clean up input document string
    doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
    doc = sub(r'<[^<>]+(>|$)', " ", doc)
    doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
    doc = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
    return [token for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if token not in stopwords]


def preprocess_semantic_similarity(documents, glove):
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


def calculate_fuzzy_similarity(query, documents):
    sim = []
    for doc in documents:
        sim.append(fuzz.token_set_ratio(query, doc))
    return sim


# Rate all possible matches
def rate_matches(responses):
    print("Preprocessing data for match rating...")

    glove = api.load("glove-wiki-gigaword-50")

    if responses[:, Person.ABOUT].tolist().count('') != len(responses[:, Person.ABOUT]):
        preprocessed_about = preprocess_semantic_similarity(responses[:, Person.ABOUT], glove)
    if responses[:, Person.WANT].tolist().count('') != len(responses[:, Person.WANT]):
        preprocessed_want = preprocess_semantic_similarity(responses[:, Person.WANT], glove)

    print("Rating all matches now...", end="\r")
    progress = [int(.1 * len(responses)), int(.2 * len(responses)), int(.3 * len(responses)),
                int(.4 * len(responses)), int(.5 * len(responses)), int(.6 * len(responses)),
                int(.7 * len(responses)), int(.8 * len(responses)), int(.9 * len(responses))]

    ratings = np.zeros((1, len(responses)))
    for r_index in range(len(responses)):
        if r_index in progress:
            percent = (progress.index(r_index) + 1) * 10
            print("Rating all matches now... " + str(percent) + "% complete", end="\r")

        curr_rating = np.zeros((len(responses)))

        # get similarities listed for other responses
        if responses[r_index, Person.MAJOR]:
            major = calculate_fuzzy_similarity(responses[r_index, Person.MAJOR], responses[:, Person.MAJOR])
            major[r_index] = 0
            major = (np.array(major) / max(major)) * Weights.MAJOR
            curr_rating += major

        if responses[r_index, Person.HOBBIES]:
            hobbies = calculate_fuzzy_similarity(responses[r_index, Person.HOBBIES], responses[:, Person.HOBBIES])
            hobbies[r_index] = 0
            hobbies = (np.array(hobbies) / max(hobbies)) * Weights.HOBBIES
            curr_rating += hobbies

        # might want to switch for tfidf for speed..
        if responses[r_index, Person.ABOUT]:
            assert(preprocessed_about)
            # about = calculate_fuzzy_similarity(responses[r_index, Person.ABOUT], responses[:, Person.ABOUT])
            about = calculate_semantic_similarity(responses[r_index, Person.ABOUT], preprocessed_about)
            about[r_index] = 0
            about = (np.array(about) / max(about)) * Weights.ABOUT
            curr_rating += about

        if responses[r_index, Person.WANT]:
            assert(preprocessed_want)
            # want = calculate_fuzzy_similarity(responses[r_index, Person.WANT], responses[:, Person.WANT])
            want = calculate_semantic_similarity(responses[r_index, Person.WANT], preprocessed_want)
            want[r_index] = 0
            want = (np.array(want) / max(want)) * Weights.WANT
            curr_rating += want

        for o_index in range(len(responses)):
            if r_index != o_index:
                response = responses[r_index]
                other = responses[o_index]

                # if response GENDER not in other PAL_GENDER or vice versa, can't match
                if (response[Person.PAL_GENDER] != "doesn't matter"
                        and other[Person.GENDER] not in response[Person.PAL_GENDER]) \
                        or (other[Person.PAL_GENDER] != "doesn't matter"
                            and response[Person.GENDER] not in other[Person.PAL_GENDER]):

                    curr_rating[o_index] = -1
                    pass

                # Year: +2 if same year, -1 if large age gap
                if response[Person.YEAR] == other[Person.YEAR]:
                    curr_rating[o_index] += Weights.YEAR
                elif (response[Person.YEAR] == "1st Year" or response[Person.YEAR] == "2nd Year") \
                        and (other[Person.YEAR] == "5th Year +" or other[Person.YEAR] == "Grad Student"):
                    curr_rating[o_index] -= Weights.YEAR

                if response[Person.ELEMENT] == other[Person.ELEMENT]:
                    curr_rating[o_index] += Weights.ELEMENT
                if response[Person.HOUSE] == other[Person.HOUSE]:
                    curr_rating[o_index] += Weights.HOUSE

        ratings = np.append(ratings, np.reshape(curr_rating, (1, len(responses))), axis=0)
    print("Rating all matches now... 100% complete")

    return ratings[1:, :]  # need to append another column of 0s


# create the final assignments
def assign_penpals(data, match_ratings):
    penpals = np.ones((len(data), 2)).astype(int)
    penpals[:] = -1

    rating_pool = [*range(len(match_ratings))]

    multiple_pals = []
    for pal in range(len(data)):
        if data[pal][Person.MULTIPLE] == "Yes":
            multiple_pals.append(pal)

    while len(rating_pool) != 0:

        # Need some matches structure.. how do I do this with multiple pals?
        # Array of ints representing match for that index, if multiple matches, list in that index.
        index_of_person = rating_pool[0]
        person = match_ratings[index_of_person]

        while person.tolist().index(max(person)) not in rating_pool \
                and person.tolist().index(max(person)) not in multiple_pals:
            person[person.tolist().index(max(person))] = 0

        index_of_pal = person.tolist().index(max(person))
        if index_of_pal in rating_pool:
            penpals[index_of_person][0] = int(index_of_pal) + 1
            rating_pool.remove(index_of_person)

            penpals[index_of_pal][0] = int(index_of_person) + 1
            rating_pool.remove(index_of_pal)
        elif index_of_pal in multiple_pals:
            penpals[index_of_person][0] = int(index_of_pal) + 1
            rating_pool.remove(index_of_person)

            penpals[index_of_pal][1] = int(index_of_person) + 1
            multiple_pals.remove(index_of_pal)
        else:
            print("the pal must have been in either rating_pool or multiple_pals")
            print("Algorithm doesn't converge..")
            rating_pool.remove(index_of_person)

    return penpals


def write_matrix_to_file(filename, list):
    with open(filename, mode='w+') as f:
        for i in list:
            for j in range(len(i)):
                if j == 0:
                    f.write(str(i[j]))
                else:
                    f.write('\t' + str(i[j]))
            f.write('\n')


def main():  # args: -[r] filename [ratings_output_file] penpals_output_file
    if len(sys.argv) != 2:
        sys.exit(1)

    data = list(csv.reader(open(sys.argv[1], "r"), delimiter="\t"))
    remove_duplicate_entries(data)

    prompts = np.array(data[0])[1:].tolist()
    prompts.insert(0, "Hash Number")
    prompts.append("1st Penpal")
    prompts.append("2nd Penpal")

    data = np.array(data)[1:, :]

    # Rate all potential Penpal matches
    ratings = rate_matches(data)
    write_matrix_to_file('ratings.tsv', ratings)

    print("Assigning penpals... ", end="")
    # Create Penpal Assignments
    penpal_assignments = np.append(data, assign_penpals(data, ratings), axis=1)[:, 1:]

    # Format Output Table: Add hash number
    penpal_assignments = np.append(np.array(range(1, len(penpal_assignments) + 1)).reshape(len(penpal_assignments), 1),
                                   penpal_assignments, axis=1)

    # Format Output Table: Add prompts
    penpal_assignments = np.append(np.array(prompts).reshape(1, 18), penpal_assignments, axis=0)

    # Write penpals to file!!
    write_matrix_to_file('penpal_matches.tsv', penpal_assignments)
    print("Done!")


if __name__ == "__main__":
    main()
