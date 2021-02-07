# UW Penpals Matchmaking Script
# 1/30/2021
# Author Kristofer Wong
import sys
import csv
import numpy as np
import json
import match_rater

def match(assignments, p0, p1, ismatched, optionsLeft):
    assignments[p0].append(p1)
    assignments[p1].append(p0)
    ismatched[p0] = True
    ismatched[p1] = True
    optionsLeft[p0] += 100
    optionsLeft[p1] += 100


def unmatch(assignments, p0, p1, ismatched, optionsLeft):
    assignments[p0].remove(p1)
    assignments[p1].remove(p0)
    ismatched[p0] = len(assignments[p0]) != 0
    ismatched[p1] = len(assignments[p1]) != 0
    optionsLeft[p0] -= 100
    optionsLeft[p1] -= 100

def notMatched(person, assignments):
    return len(assignments[person]) == 0


def prefers(ratings, person, pal0, pal1):
    return ratings[person][pal0] > ratings[person][pal1]


# Never ever ever getting back together
def taylorSwift(ratings, p0, p1, optionsLeft):
    # print("tswift: " + str(p0) + " " + str(p1))
    ratings[p0][p1] = -1
    ratings[p1][p0] = -1
    optionsLeft[p0] -= 1
    optionsLeft[p1] -= 1


def otherOptions(ratings, assignments, max_pals, ismatched, person):
    prob_options = 0
    total_options = 0
    for option in range(len(ratings[person].tolist())):
        if ratings[person][option] != -1:
            total_options += 1
            if ismatched[option] and max_pals[option] == len(assignments[option]):
                prob_options += 1

    # print(prob_options, total_options)
    if prob_options == total_options:
        return False
    else:
        return ratings[person].tolist().count(-1) != len(ratings[person])


# Assigns all penpals. 
# Uses a modified version of the Gale-Shapley stable marriage algorithm
# Does not guarantee stable matching, but given large enough MAX_PALS 
# parameter will match everyone together.
# TODO: Could be more efficient with better style. Will fix given free time
def assign_penpals(data, match_ratings, MAX_PALS, MULTIPLE_PALS):
    # set up max_pals array
    max_pals = []
    for person in data:
        if person[MULTIPLE_PALS] == "Yes":
            max_pals.append(MAX_PALS)
        else:
            max_pals.append(1)

    ismatched = [False] * len(data)

    optionsLeft = []
    for person in match_ratings:
        optionsLeft.append(len(person) - person.tolist().count(-1))

    penpal_assignments = pd = [[] for _ in range(len(data))]

    while ismatched.count(False) > 0:
        # p is index of person we're finding match for
        p = optionsLeft.index(min(optionsLeft))
        p_prefs = match_ratings[p].tolist()
        pal = p_prefs.index(max(p_prefs))

        if match_ratings[p][pal] < 0:
            print("bad match: " + str(p) + " " + str(pal) + " " + str(match_ratings[p][pal]))
            optionsLeft[p] += 100
            ismatched[p] = True
        if not ismatched[pal] or max_pals[pal] > len(penpal_assignments[pal]):
            match(penpal_assignments, p, pal, ismatched, optionsLeft)
        else:
            for altpal in penpal_assignments[pal]:
                if prefers(match_ratings, pal, p, altpal) and otherOptions(match_ratings, penpal_assignments, max_pals, ismatched, altpal):
                    unmatch(penpal_assignments, pal, altpal, ismatched, optionsLeft)
                    taylorSwift(match_ratings, pal, altpal, optionsLeft)
                else:
                    taylorSwift(match_ratings, p, pal, optionsLeft)

    return penpal_assignments


# Formats input data to get rid of duplicate entries
def remove_duplicate_entries(responses, NAME, ADDRESS):
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


# Writes a matrix to given file name. `list` may be list of lists or np matrix
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

    # About data inputting
    #   mc_attr, semantic_attr, and fuzzy_attr come from attributes.json
    #   each will be an array with the indices into a response to get to x attribute
    #   attr_weights is an array of arrays, [[mc_weights], [fuzzy_weights], [semantic_weights]]
    
    # Process input attributes json file:
    with open(sys.argv[2], "r") as f:
        attributes = json.load(f)

    NAME = attributes["NAME"][0]
    ADDRESS_0 = attributes["ADDRESS_0"][0]
    ADDRESS_1 = attributes["ADDRESS_1"][0]
    EMAIL_0 = attributes["EMAIL_0"][0]
    EMAIL_1 = attributes["NAME"][0]
    MULTIPLE_PALS = attributes["MULTIPLE_PALS"][0]
    PRONOUNS = attributes["PRONOUNS"][0]

    # Preprocess the input data from the Google Form
    data = list(csv.reader(open(sys.argv[1], "r"), delimiter="\t"))
    remove_duplicate_entries(data, NAME + 1, ADDRESS_0 + 1)

    # Get rid of prompts from input data
    data = np.array(data)[1:, 1:]

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
    print("Rating all poassible matches...")
    ratings = match_rater.rate_matches(data, mc_attr, fuzzy_attr, semantic_attr, attr_weights)
    write_matrix_to_file('ratings.tsv', ratings)

    # Create Penpal Assignments
    print("Assigning penpals... ")
    assignments = assign_penpals(data, ratings, 6, MULTIPLE_PALS)
    
    # Format output table & write to penpal_matches.tsv
    print("All penpals assigned.. Writing to output")
    for p_index in range(len(assignments)):
        assignments[p_index].insert(0, p_index)
        assignments[p_index].insert(1, data[p_index][0])
        assignments[p_index].insert(2, data[p_index][7])
        assignments[p_index].insert(3, data[p_index][9])
        assignments[p_index].insert(4, data[p_index][10])
        if len(assignments[p_index]) == 5:
            assignments[p_index].append("NO MATCH")

    # Write penpals to file!!
    write_matrix_to_file('penpal_matches.tsv', assignments)
    print("Done!")


if __name__ == "__main__":
    main()
