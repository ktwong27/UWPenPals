# UW Penpals Matchmaking Script
# 1/30/2021
# Author Kristofer Wong

import sys
import csv
import numpy as np
import json
import match_rater

NAME = 0
ADDRESS = 0
EMAIL_0 = 0
EMAIL_1 = 0
MULTIPLE_PALS = 0
PRONOUNS = 0

def remove_duplicate_entries(responses):
    # Get rid of duplicates
    names = set()
    addresses = set()

    # reversed so we get rid of first submission.
    # people tend to submit multiple times to change something
    for response in reversed(responses):
        if response[NAME] in names \
                and response[ADDRESS] in addresses:
            responses.remove(response)
        names.add(response[NAME])
        addresses.add(response[ADDRESS])


# create the final assignments 
# TODO: Change this to adapted Stable Roommates problem with replacement for those who are ok with multiple pals
def assign_penpals(data, match_ratings):
    penpals = np.ones((len(data), 2)).astype(int)
    penpals[:] = -1

    rating_pool = [*range(len(match_ratings))]

    multiple_pals = []
    for pal in range(len(data)):
        if data[pal][MULTIPLE] == "Yes":
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


def main():  # args: input_filename attributes_filename
    if len(sys.argv) != 3:
        print("Usage: python3 matchmaker.py input_file attributes_json")
        sys.exit(1)

    # Preprocess the input data from the Google Form
    data = list(csv.reader(open(sys.argv[1], "r"), delimiter="\t"))
    remove_duplicate_entries(data)

    # Set up prompts for output
    prompts = np.array(data[0])[1:].tolist()
    prompts.insert(0, "Hash Number")
    prompts.append("1st Penpal")
    prompts.append("2nd Penpal")

    # Get rid of prompts from input data
    data = np.array(data)[1:, :]


    # About data inputting
    #   mc_attr, semantic_attr, and fuzzy_attr come from attributes.json
    #   each will be an array with the indices into a response to get to x attribute
    #   attr_weights is an array of arrays, [[mc_weights], [fuzzy_weights], [semantic_weights]]
    # Process input attributes json file:
    with open(sys.argv[2], "r") as f:
        attributes = json.load(f)

    NAME = attributes["NAME"][0]
    ADDRESS = attributes["ADDRESS"][0]
    EMAIL_0 = attributes["EMAIL_0"][0]
    EMAIL_1 = attributes["NAME"][0]
    MULTIPLE_PALS = attributes["MULTIPLE_PALS"][0]
    PRONOUNS = attributes["PRONOUNS"][0]

    mc_attr = []
    fuzzy_attr = []
    semantic_attr = []
    attr_weights = [[], [], []]
    for attribute in attributes:
        attr_data = attributes[attribute]
        if attr_data[1] == 1:
            mc_attr.append(attr_data[0])
            attr_weights[0].append(attr_data[2])
        elif attr_data[1] == 2:
            fuzzy_attr.append(attr_data[0])
            attr_weights[1].append(attr_data[2])
        elif attr_data[1] == 3:
            semantic_attr.append(attr_data[0])
            attr_weights[2].append(attr_data[2])

    # Rate all potential Penpal matches
    ratings = match_rater.rate_matches(data, mc_attr, fuzzy_attr, semantic_attr, attr_weights)

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
