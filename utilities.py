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

def wta(items):
    """ Winner-take-all: pick the largest item from the list """
    maxweight = max(items)
    candidates = []
    for i in range(len(items)):
        if items[i] == maxweight:
            candidates.append(i)
    return random.choice(candidates)