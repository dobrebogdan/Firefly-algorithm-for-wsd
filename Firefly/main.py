# TODO: Index usage in local search
# TODO: Decide on the good lesk implementation
# TODO: Evaluate on all 3 corpuses if possible
# TODO: Bugfixes + performance issues

import nltk
import random
from os import listdir
from os.path import isfile, join
nltk.download('brown')
nltk.download('semcor')
nltk.download('senseval')
nltk.download('wordnet_ic')
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet as wn, wordnet_ic
from nltk.corpus import semcor, senseval
from text_utils import clean_text, clean_and_tokenize, phrase_replace, phrase_contains
from scipy.spatial import distance
import numpy as np
import xml.etree.ElementTree as ET

# Constants

W = 5
# HFA parameters
SWARM_SIZE = 100
E = 2.72
MAX_ITER = 3
ALPHA = 0.2
GAMMA = 1
CORPUS = semcor
CORPUS_IC = wordnet_ic.ic(f'./ic-semcor.dat')

# Local search parameters
LR = 0.15
L_FA = 17000
MAX_CYCLES = 30000

ps = PorterStemmer()
wl = WordNetLemmatizer()

# Reading
semcor_path = '../../semcor/data'

sentences = []
true_fireflies = []
covered_tokens = 0
total_tokens = 0


xml_files = sorted([f for f in listdir(semcor_path) if isfile(join(semcor_path, f))])[:1]
for xml_file in xml_files:
    tree = ET.parse(f'{semcor_path}/{xml_file}')
    root = tree.getroot()[0]
    for paragraph_tag in root:
        for sentence_tag in paragraph_tag:
            sentence = []
            firefly = []
            for wf_tag in sentence_tag:
                total_tokens += 1
                try:
                    if ('wnsn' in wf_tag.attrib.keys()) and int(wf_tag.attrib['wnsn']) != 0:
                        covered_tokens += 1
                        wnsn_pos = int(wf_tag.attrib['wnsn']) - 1
                        words = clean_text(wf_tag.text).split(' ')
                        for word in words:
                            if len(wn.synsets(word)) <= wnsn_pos:
                                continue
                            firefly.append(wn.synsets(word)[wnsn_pos])
                            sentence.append(word)
                except:
                    pass

            sentences.append(sentence)
            true_fireflies.append(firefly)


print(sentences[:2])

# Algorithm

def get_context(curr_synset):
    context = [curr_synset]
    context.extend(curr_synset.hypernyms())
    context.extend(curr_synset.hyponyms())
    context.extend(curr_synset.also_sees())
    context.extend(curr_synset.member_holonyms())
    context.extend(curr_synset.substance_holonyms())
    context.extend(curr_synset.part_holonyms())
    context.extend(curr_synset.member_meronyms())
    context.extend(curr_synset.substance_meronyms())
    context.extend(curr_synset.part_meronyms())
    context.extend(curr_synset.attributes())
    context.extend(curr_synset.similar_tos())
    return list(set(context))


def overlap_score(synset1, synset2):
    score = 0
    gloss1 = synset1.definition()
    gloss2 = synset2.definition()
    tokens1 = clean_and_tokenize(gloss1)
    tokens2 = clean_and_tokenize(gloss2)
    tokens1 = [ps.stem(token) for token in tokens1]
    tokens2 = [ps.stem(token) for token in tokens2]
    candidate_phrases = []

    # Since we are looking for common phrases, considering just the ones in gloss1 as candidates is enough
    for size in range(len(tokens1), 0, -1):
        for i in range(0, len(tokens1) - size + 1):
            candidate_phrase = tokens1[i:(i+size)]
            candidate_phrases.append(candidate_phrase)

    for candidate_phrase in candidate_phrases:
        if phrase_contains(tokens1, candidate_phrase) and phrase_contains(tokens2, candidate_phrase):
            score += len(candidate_phrase) * len(candidate_phrase)
            tokens1 = phrase_replace(tokens1, candidate_phrase)
            tokens2 = phrase_replace(tokens2, candidate_phrase)
    return score


def extended_lesk_score(synset1, synset2):
    context1 = get_context(synset1)
    context2 = get_context(synset2)
    score = 0
    for context_synset1 in context1:
        for context_synset2 in context2:
            score += overlap_score(context_synset1, context_synset2)
    return score


def IC(synset1, synset2):
    return synset1.res_similarity(synset2, CORPUS_IC)


def synset_sense_score(synset1, synset2):
    ic_score = 0.0
    try:
        ic_score = IC(synset1, synset2)
    except:
        pass
    return extended_lesk_score(synset1, synset2) + ic_score


def sentence_sense_score(synsets):
    score = 0
    for i in range(0, len(synsets) - W):
        window = synsets[i:(i+W)]
        for j in range(0, W):
            for k in range(j+1, W):
                score += synset_sense_score(window[j], window[k])
    return score



def synset_to_coord(synset, word):
    return wn.synsets(word).index(synset)


def sentence_to_coords(synsets, words):
    return [float(synset_to_coord(synset, word)) for (synset, word) in zip(synsets, words)]


def coord_to_synset(coord, word):
    if coord < 0:
        coord *= -1
    coord = int(coord)
    return wn.synsets(word)[coord]


def coords_to_sentence(coords, words):
    return [coord_to_synset(coord, word) for (coord, word) in zip(coords, words)]


def move_fireflies(fireflies, firefly_intensities, words):
    coords = np.array([sentence_to_coords(firefly, words) for firefly in fireflies])
    for i in range(0, len(coords)):
        for j in range(0, len(coords)):
            if i != j:
                if firefly_intensities[j] > firefly_intensities[i]:
                    r = distance.euclidean(coords[i], coords[j])
                    beta_r = 1.0 * firefly_intensities[j] * (E ** (-GAMMA * r * r))
                    coords[i] = coords[i] + beta_r * (coords[j] - coords[i]) + ALPHA * (random.uniform(0, 1) - 0.5)

    fireflies = [coords_to_sentence(coord, words) for coord in coords]
    return fireflies



def local_search(initial_firefly, initial_firefly_intensity, words):
    firefly_list = [(initial_firefly_intensity, initial_firefly)]
    for x in range(0, MAX_CYCLES):
        current_firefly = initial_firefly.copy()
        idx1 = random.randint(0, len(words) - 1)
        idx2 = random.randint(0, len(words) - 1)
        idx1_synsets = wn.synsets(words[idx1])
        idx2_synsets = wn.synsets(words[idx2])
        idx1_pos = random.randint(0, len(idx1_synsets) - 1)
        idx2_pos = random.randint(0, len(idx2_synsets) - 1)
        current_firefly[idx1] = idx1_synsets[idx1_pos]
        current_firefly[idx2] = idx2_synsets[idx2_pos]
        current_firefly_intensity = sentence_sense_score(current_firefly)
        if current_firefly_intensity > firefly_list[0][0]:
            new_firefly_list = [(current_firefly_intensity, current_firefly)]
            new_firefly_list.extend(firefly_list)
            new_firefly_list.pop()
            firefly_list = new_firefly_list
        else:
            if len(firefly_list) < L_FA:
                firefly_list.append((initial_firefly_intensity, initial_firefly))
    return firefly_list[0]


# Creating a synset frequencies dictionary for the corpus to calculate IC


print(f'COVERAGE:{covered_tokens / total_tokens}')
# Getting and parsing sents
total_tokens_nr = 0
covered_tokens_nr = 0
sentence_no = 0

tp = 0
fp = 0
tn = 0
fn = 0
for sentence in sentences[:4]:
    sentence_no += 1
    tokens = sentence
    synsets_for_tokens = [wn.synsets(token) for token in tokens]

    # Deploy swarm of fireflies
    fireflies = []
    firefly_intensities = []
    for _ in range(0, SWARM_SIZE):
        curr_firefly_synsets = []
        for synsets_for_token in synsets_for_tokens:
            synsets_num =len(synsets_for_token)
            curr_firefly_synsets.append(synsets_for_token[random.randint(0, synsets_num-1)])
        fireflies.append(curr_firefly_synsets)
        firefly_intensities.append(sentence_sense_score(curr_firefly_synsets))

    global_best_firefly = fireflies[0]
    global_best_intensity = firefly_intensities[0]
    # Start iterations
    for iter_no in range(0, MAX_ITER):
        print(f'Iteration number: {iter_no}')
        fireflies = move_fireflies(fireflies, firefly_intensities, tokens)
        firefly_intensities = [sentence_sense_score(firefly) for firefly in fireflies]
        intensities_and_positions = zip(firefly_intensities, fireflies)
        (top_intensity, top_firefly) = sorted(intensities_and_positions, reverse=True)[0]
        print(f'Before local search')
        (best_intensity, best_firefly) = local_search(top_firefly, top_intensity, tokens)
        print(f'After local search')
        if best_intensity > global_best_intensity:
            global_best_intensity = best_intensity
            global_best_firefly = best_firefly

    true_firefly = true_fireflies[sentence_no-1]
    print(f'RESULTS FOR SENTENCE {sentence_no}')
    print(global_best_firefly)
    print(true_firefly)
    print(tokens)

    for i in range(0, len(true_firefly)):
        if true_firefly[i] == global_best_firefly[i]:
            tp += 1
            tn += len(wn.synsets(tokens[i])) - 1
        else:
            fp += 1
            fn += 1
            tn += len(wn.synsets(tokens[i])) - 1

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = (2 * precision * recall) / (precision + recall)
print(f"Precision: {precision}\nRecall: {recall}\nF1 Score: {f1_score}")
