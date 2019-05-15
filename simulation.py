import random
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf')

from math import log, log1p, exp
from scipy.special import logsumexp

from copy import deepcopy

import utilities

""" Parameters """
noise = 0.05
num_signals = 2
context = [0.5, 0.9] # where are the referents
perspectives = [0, 1]
languages = [[[1, 1], [1, 1]], [[1, 1], [1, 0]], [[1, 1], [0, 1]], [[1, 1], [0, 0]], [[1, 0], [1, 1]], [[1, 0], [1, 0]], [[1, 0], [0, 1]], [[1, 0], [0, 0]], [[0, 1], [1, 1]], [[0, 1], [1, 0]], [[0, 1], [0, 1]], [[0, 1], [0, 0]], [[0, 0], [1, 1]], [[0, 0], [1, 0]], [[0, 0], [0, 1]]]
# generate list of language-perspective pairs
lp_pairs = []
for l in languages:
    for p in perspectives:
        lp_pairs.append([l, p])

# def reception_weights(system, signal):
#     weights = []
#     for row in system:
#         weights.append(row[signal])
#     return weights

# def communicate(speaker_system, hearer_system, meaning):
#     speaker_signal = wta(speaker_system[meaning])
#     hearer_meaning = wta(reception_weights(hearer_system, speaker_signal))
#     if meaning == hearer_meaning: 
#         return 1
#     else: 
#         return 0

# def learn(system, meaning, signal):
#     system[meaning][signal] += 1

# def train(system, ms_pair_list):
#     for pair in ms_pair_list:
#         learn(system, pair[0], pair[1])

def update_posterior(posterior, meaning, signal):
    in_language = log(1 - noise)
    out_of_language = log(noise / (num_signals - 1))
    new_posterior = []
    for i in range(len(posterior)):
        if (meaning, signal) in languages[i]:
            new_posterior.append(posterior[i] + in_language)
        else:
            new_posterior.append(posterior[i] + out_of_language)
    return utilities.normalize_logprobs(new_posterior)

def produce(system, no_productions):
    ms_pairs = []
    for n in range(no_productions):
        speaker = system
        # TODO: replace this with speaker picking based on perspective and context
        meaning = random.randrange(len(speaker)) 
        signal = utilities.wta(speaker[meaning])
        ms_pairs.append([meaning, signal])  
    return ms_pairs

speaker = lp_pairs[5]
print(produce(speaker[0], 5))