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
num_signals = 3
meanings = [0, 1, 2]
signals = ['a', 'b', 'c']
# num_signals = 2
# meanings= [0, 1]
# signals = ['a', 'b']
perspectives = [0, 1]
# languages = [[[1, 1], [1, 1]], [[1, 1], [1, 0]], [[1, 1], [0, 1]], [[1, 0], [1, 1]], [[1, 0], [1, 0]], [[1, 0], [0, 1]], [[0, 1], [1, 1]], [[0, 1], [1, 0]], [[0, 1], [0, 1]]]
languages = []
for i in range(pow(2, 9), 0, -1):
    string = str(format(i, 'b'))
    if len(string) >= 7:
        while len(string) < 9:
            string = '0' + string
        if string[-3:] != '000' and string[3:-3] != '000':
            languages.append([[string[0],string[1],string[2]], [string[3],string[4],string[5]], [string[6],string[7],string[8]]])
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
    new_posterior = []
    for i in range(len(posterior)): # for each hypothesis
        language = lp_pairs[i][0]
        ref_distribution = calc_mental_state(lp_pairs[i][1], context)
        marginalize = []
        for meaning in meanings:
            signals_for_r = [signals[s] for s in range(len(meanings)) if language[meaning][s] == '1']
            num_signals_for_r = sum([int(i) for i in language[meaning]])
            if signal in signals_for_r:
                in_language = log((1 - noise) / num_signals_for_r)
                marginalize.append(ref_distribution[meaning] + in_language)
            else:
                out_of_language = log(noise / (num_signals - num_signals_for_r))
                marginalize.append(ref_distribution[meaning] + out_of_language)
        # print(exp(logsumexp(marginalize)))
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

speaker1 = lp_pairs[188]
speaker2 = lp_pairs[0]
speaker3 = lp_pairs[10]
# speaker = lp_pairs[5]
# print(lp_pairs.index([['100', '010', '001'], 0]))
post = priors
contexts = [[random.random(), random.random(), random.random()] for i in range(500)]
posterior_list1 = [exp(post[5])]
posterior_list2 = [exp(post[5])]
posterior_list3 = [exp(post[5])]
for i in range(300):
    d = produce(speaker1, contexts[i])
    post = update_posterior(post, d[0], d[1])
    posterior_list1.append(exp(post[188]))
# for i in range(100):
#     d = produce(speaker2, contexts[i])
#     post = update_posterior(post, d[0], d[1])
#     posterior_list2.append(exp(post[0]))
# for i in range(100):
#     d = produce(speaker3, contexts[i])
#     post = update_posterior(post, d[0], d[1])
#     posterior_list3.append(exp(post[10]))
plt.plot(posterior_list1)
# plt.plot(posterior_list2)
# plt.plot(posterior_list3)
plt.savefig('plot.png')