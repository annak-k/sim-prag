import numpy as np
from math import log

import utilities

def generate_languages():
    """ Generate 3x3 lexicon matrices, only if every meaning has at least one signal """
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

def generate_hypotheses(perspectives, p_learner, bias="unbiased", pragmatic_levels=None):
    """ Generate list of language-perspective hypotheses or 
        language-perspective-pragmatic-level hypotheses and generate their priors """
    languages = generate_languages()
    hypotheses = []
    priors = []
    if pragmatic_levels != None:
        for pl in pragmatic_levels:
            for p in perspectives:
                for l in languages:
                    hypotheses.append([l, p, pl])
                    
                    if bias == "unbiased":
                        p_prior = log(1/len(perspectives))
                    elif bias == "egocentric":
                        p_prior = log(0.9) if p == p_learner else log(0.1)
                    priors.append(p_prior + log(1/len(languages)) + log(1/len(pragmatic_levels)))
    else:
        for p in perspectives:
            for l in languages:
                hypotheses.append([l, p])

                if bias == "unbiased":
                    p_prior = log(1/len(perspectives))
                elif bias == "egocentric":
                    p_prior = log(0.9) if p == p_learner else log(0.1)
                priors.append(p_prior + log(1/len(languages)))
    return [np.array(hypotheses), np.array(priors)]