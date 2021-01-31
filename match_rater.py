import short_response_parsing

def rate_mc(matrix_ratings, responses, mc_attributes, attribute_weights):
    # Function requires the following indexes correspond in mc_attributes and attribute_weights
    YEAR = 0
    GENDER = 1
    PAL_GENDER = 2

    print(matrix_ratings)

    for response_idx in range(len(responses)):
        curr_rating = matrix_ratings[response_idx]

        for other_idx in range(len(responses)):
            if response_idx == other_idx:
                continue

            response = responses[response_idx]
            other = responses[other_idx]

            # Specific attributes (Gender, Year)
            # if response GENDER not in other PAL_GENDER or vice versa, can't match
            if (response[PAL_GENDER] != "doesn't matter"
                    and other[mc_attributes[GENDER]] not in response[mc_attributes[PAL_GENDER]]) \
                    or (other[PAL_GENDER] != "doesn't matter"
                        and response[mc_attributes[GENDER]] not in other[mc_attributes[PAL_GENDER]]):
                curr_rating[other_idx] = -1
                continue
                pass

            # Year: +2 if same year, don't match if large gap
            if response[mc_attributes[YEAR]] == other[mc_attributes[YEAR]]:
                curr_rating[other_idx] += attribute_weights[YEAR]
            elif (response[mc_attributes[YEAR]] == "1st Year" or response[mc_attributes[YEAR]] == "2nd Year") \
                    and (other[mc_attributes[YEAR]] == "5th Year +" or other[mc_attributes[YEAR]] == "Grad Student"):
                curr_rating[other_idx] = -1
                continue


            # General attributes
            for attribute in range(len(mc_attributes)):
                if response[mc_attributes[attribute]] == other[mc_attributes[attribute]]:
                    curr_rating[other_idx] += attribute_weights[attribute]

    return matrix_ratings


def rate_matches(responses, mc_attributes, fuzzy_attributes, semantic_attributes, attribute_weights):

    # print("Rating all matches now...", end="\r")
    # progress = [int(.1 * len(responses)), int(.2 * len(responses)), int(.3 * len(responses)),
    #             int(.4 * len(responses)), int(.5 * len(responses)), int(.6 * len(responses)),
    #             int(.7 * len(responses)), int(.8 * len(responses)), int(.9 * len(responses))]

    # Rate all matches based on short response questions
    matrix_ratings = short_response_parsing.rate_short_responses(responses, fuzzy_attributes, semantic_attributes, attribute_weights[1:])

    # Adjust rating for all matches' multiple choice characteristics
    matrix_ratings = rate_mc(matrix_ratings, responses, mc_attributes, attribute_weights[0])

    return matrix_ratings

