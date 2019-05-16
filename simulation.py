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
meanings = [0, 1]
signals = ['a', 'b']
context1 = [0.5, 0.9] # where are the referents
perspectives = [0, 1]
languages = [[[1, 1], [1, 1]], [[1, 1], [1, 0]], [[1, 1], [0, 1]], [[1, 0], [1, 1]], [[1, 0], [1, 0]], [[1, 0], [0, 1]], [[0, 1], [1, 1]], [[0, 1], [1, 0]], [[0, 1], [0, 1]]]
# generate list of language-perspective pairs
lp_pairs = []
for p in perspectives:
    for l in languages:
        lp_pairs.append([l, p])
priors = []
for p in perspectives:
    for l in languages:
        priors.append(log(1/len(perspectives)) + log(1/len(languages)))

def sample(posterior):
    return meanings[utilities.log_roulette_wheel(posterior)]

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
""" Given speaker's perspective and the context,
    compute a probability distribution over the referents
    of how likely the speaker is to speak about each referent
    p. 88 Equation 3.1 """
def calc_mental_state(perspective, context):
    distribution = []
    for o in context:
        distribution.append(log(1 - abs(perspective - o)))
    return utilities.normalize_logprobs(distribution)

# TODO: Make this work
def update_posterior(posterior, signal, context):
    in_language = log(1 - noise)
    out_of_language = log(noise / (num_signals - 1))
    new_posterior = []
    for i in range(len(posterior)): # for each hypothesis
        ref_distribution = calc_mental_state(lp_pairs[i][1], context)
        marginalize = []
        for meaning in meanings:
            if (meaning, signal) in lp_pairs[i][0]:
                marginalize.append(ref_distribution[meaning] + in_language)
            else:
                marginalize.append(ref_distribution[meaning] + out_of_language)
        # print(marginalize)
        new_posterior.append(posterior[i] + logsumexp(marginalize))
    return utilities.normalize_logprobs(new_posterior)

def produce(system, context):
    language = system[0]
    meaning = sample(calc_mental_state(system[1], context))
    signal = signals[utilities.wta(language[meaning])]
    if random.random() < noise:
        other_signals = deepcopy(signals)
        other_signals.remove(signal)
        signal = random.choice(other_signals)
    return [signal, context]

speaker = lp_pairs[5]
post = priors
contexts = [[0.1, 0.5], [0.7, 0.3], [0.9, 0.1], [0.1, 0.9], [0.5, 0.5], [0.1, 0.5], [0.7, 0.3], [0.9, 0.1], [0.1, 0.9], [0.5, 0.5]]
for i in range(10):
    d = produce(speaker, contexts[i])
    post = update_posterior(post, d[0], d[1])
    # print(post)
    print(exp(post[5]))