from math import log, log1p, exp
from scipy.special import logsumexp
import random
import numpy as np

def log_subtract(x,y):
    return x + log1p(-exp(y - x))

def normalize_probs(probs):
    total = sum(probs) #calculates the summed log probabilities
    normedprobs = []
    for p in probs:
        normedprobs.append(p / total) #normalise - subtracting in the log domain equivalent to divising in the normal domain
    return normedprobs

def normalize_logprobs(logprobs):
    logtotal = logsumexp(logprobs) #calculates the summed log probabilities
    normedlogs = []
    for logp in logprobs:
        normedlogs.append(logp - logtotal) #normalise - subtracting in the log domain equivalent to divising in the normal domain
    return np.array(normedlogs)

def log_roulette_wheel(normedlogs):
    r=log(random.random()) #generate a random number in [0,1), then convert to log
    accumulator = normedlogs[0]
    for i in range(len(normedlogs)):
        if r < accumulator:
            return i
        accumulator = logsumexp([accumulator, normedlogs[i + 1]])

""" Winner-take-all: pick the largest item from the list """
def wta(items):
    maxweight = max(items)
    candidates = []
    for i in range(len(items)):
        if items[i] == maxweight:
            candidates.append(i)
    return random.choice(candidates)

""" Generate 3x3 lexicon matrices, only if every meaning has at least one signal """
def generate_languages():
    languages = []
    for i in range(pow(2, 9), 0, -1):
        string = str(format(i, 'b'))
        if len(string) >= 7: # eliminate strings that don't have a signal for meaning 0
            while len(string) < 9:
                string = '0' + string # pad with 0s
            # eliminate strings that don't have a signal for meanings 1 and 2
            if string[-3:] != '000' and string[3:-3] != '000':
                # create matrix
                languages.append([[string[0],string[1],string[2]],
                    [string[3],string[4],string[5]],
                    [string[6],string[7],string[8]]])
    return np.array(languages)