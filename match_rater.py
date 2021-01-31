import short_response_parsing
import numpy as np

def rate_short_responses(responses, fuzzy_attributes, semantic_attributes, attribute_weights):
    # glove = api.load("glove-wiki-gigaword-50")
    # short_response_parsing.nltk.download('stopwords') # TODO: figure out why this fails

    ratings_matrix = np.zeros((1, len(responses)))
    fuzzy_weights = attribute_weights[0]
    semantic_weights = attribute_weights[1]
    print("Rating short response questions... ", end="\r")
    progress = [int(.1 * len(responses)), int(.2 * len(responses)), int(.3 * len(responses)),
                    int(.4 * len(responses)), int(.5 * len(responses)), int(.6 * len(responses)),
                    int(.7 * len(responses)), int(.8 * len(responses)), int(.9 * len(responses))]

    for r_index in range(len(responses)):
        if r_index in progress:
            percent = (progress.index(r_index) + 1) * 10
            print("Rating short response questions... " + str(percent) + "% complete", end="\r")

        curr_rating = np.zeros((len(responses)))
        # Fuzzy simularity calculations used for simple short answer questions "fuzzy_attributes" 
        for attribute in range(len(fuzzy_attributes)):
            # Don't process if this person has no response for this attribute
            if responses[r_index, fuzzy_attributes[attribute]]:
                attr_rating = short_response_parsing.calculate_fuzzy_similarity(responses[r_index, fuzzy_attributes[attribute]], responses[:, fuzzy_attributes[attribute]])
                # Don't want match with self
                attr_rating[r_index] = 0
                # Normalize fuzzy calculations to proportion of attribute weight
                if (max(attr_rating) == 0):
                    attr_rating = (np.array(attr_rating) / 1) * fuzzy_weights[attribute]
                else:
                    attr_rating = (np.array(attr_rating) / max(attr_rating)) * fuzzy_weights[attribute]
                curr_rating += attr_rating
        # Use TFIDF similarity calculations for comparing responses to open ended short answer questions
        # TODO: Not working right now - SSL issues? May need to reinstall python
        for attribute in range(len(semantic_attributes)):
            if responses[r_index, attribute]: #if preprocessed and responses[r_index, attribute]:
                attr_rating = short_response_parsing.calculate_tfidf_similarity(responses[r_index, attribute], responses[:, attribute])
                attr_rating[r_index] = 0
                if (max(attr_rating) == 0):
                    attr_rating = (np.array(attr_rating) / 1) * fuzzy_weights[attribute]
                else:
                    attr_rating = (np.array(attr_rating) / max(attr_rating)) * semantic_weights[attribute]
                curr_rating += attr_rating

        ratings_matrix = np.append(ratings_matrix, np.reshape(curr_rating, (1, len(responses))), axis=0)
    print("Rating short response questions... 100% complete")

    return ratings_matrix[1:, :]


def rate_mc(matrix_ratings, responses, mc_attributes, attribute_weights):
    # Function requires the following indexes correspond in mc_attributes and attribute_weights
    YEAR = 0
    GENDER = 1
    PAL_GENDER = 2

    print("Rating multiple choice questions... ", end="\r")
    progress = [int(.1 * len(responses)), int(.2 * len(responses)), int(.3 * len(responses)),
                    int(.4 * len(responses)), int(.5 * len(responses)), int(.6 * len(responses)),
                    int(.7 * len(responses)), int(.8 * len(responses)), int(.9 * len(responses))]

    for response_idx in range(len(responses)):
        if response_idx in progress:
            percent = (progress.index(response_idx) + 1) * 10
            print("Rating multiple choice questions... " + str(percent) + "% complete", end="\r")

        curr_rating = matrix_ratings[response_idx]

        for other_idx in range(len(responses)):
            if response_idx == other_idx:
                continue

            response = responses[response_idx]
            other = responses[other_idx]

            # Specific attributes (Gender, Year)
            # if response GENDER not in other PAL_GENDER or vice versa, can't match
            if (response[mc_attributes[PAL_GENDER]] != "Doesn't matter"
                    and other[mc_attributes[GENDER]] not in response[mc_attributes[PAL_GENDER]]) \
                    or (other[mc_attributes[PAL_GENDER]] != "Doesn't matter"
                        and response[mc_attributes[GENDER]] not in other[mc_attributes[PAL_GENDER]]):
                curr_rating[other_idx] = -1
                continue

            # Year: +2 if same year, don't match if large gap
            if response[mc_attributes[YEAR]] == other[mc_attributes[YEAR]]:
                curr_rating[other_idx] += attribute_weights[YEAR]
            elif (response[mc_attributes[YEAR]] == "1st Year" or response[mc_attributes[YEAR]] == "2nd Year") \
                    and (other[mc_attributes[YEAR]] == "5th Year +" or other[mc_attributes[YEAR]] == "Grad Student"):
                curr_rating[other_idx] = -1
                print("Age difference: ", response[0], other[0])
                continue


            # General attributes
            for attribute in range(len(mc_attributes)):
                if response[mc_attributes[attribute]] == other[mc_attributes[attribute]]:
                    curr_rating[other_idx] += attribute_weights[attribute]

    print("Rating multiple choice questions... 100% complete")
    return matrix_ratings


def rate_matches(responses, mc_attributes, fuzzy_attributes, semantic_attributes, attribute_weights):
    # Rate all matches based on short response questions
    matrix_ratings = rate_short_responses(responses, fuzzy_attributes, semantic_attributes, attribute_weights[1:])
    
    # Adjust rating for all matches' multiple choice characteristics
    matrix_ratings = rate_mc(matrix_ratings, responses, mc_attributes, attribute_weights[0])

    return matrix_ratings

