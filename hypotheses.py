import numpy as np
from math import log

import utilities

def generate_languages(size):
    """ Generate sizeXsize lexicon matrices, only if every meaning has at least one signal """
    languages = []
    if size == 3:
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
    elif size == 2:
        for i in range(pow(2, 4)-1, 0, -1):
            string = str(format(i, 'b'))
            while len(string) < 4:
                string = '0' + string # pad with 0s
            if string[:2] != '00' and string[2:] != '00':
                languages.append([[string[0],string[1]],
                                [string[2],string[3]]])
    else:
        print("Wrong language size")
        return None
            
    return np.array(languages)

def generate_hypotheses(lang_size, perspectives, p_learner, bias="unbiased", pragmatic_levels=None):
    """ Generate list of language-perspective hypotheses or 
        language-perspective-pragmatic-level hypotheses and generate their priors """
    languages = generate_languages(lang_size)
    if pragmatic_levels != None:
        hypotheses = np.zeros((len(pragmatic_levels) * len(perspectives) * len(languages), 3)).astype(int)
        priors = np.zeros(len(pragmatic_levels) * len(perspectives) * len(languages))
        count = 0
        for pl in range(len(pragmatic_levels)):
            for p in range(len(perspectives)):
                for l in range(len(languages)):
                    hypotheses[count] = [l, p, pl]
                    
                    if bias == "unbiased":
                        p_prior = log(1/len(perspectives))
                    elif bias == "egocentric":
                        p_prior = log(0.9) if p == p_learner else log(0.1)
                    priors[count] = p_prior + log(1/len(languages)) + log(1/len(pragmatic_levels))
                    count += 1
    else:
        hypotheses = np.zeros((len(perspectives) * len(languages), 2)).astype(int)
        priors = np.zeros(len(perspectives) * len(languages))
        count = 0
        for p in range(len(perspectives)):
            for l in range(len(languages)):
                hypotheses[count] = [l, p]
                
                if bias == "unbiased":
                    p_prior = log(1/len(perspectives))
                elif bias == "egocentric":
                    p_prior = log(0.9) if p == p_learner else log(0.1)
                priors[count] = p_prior + log(1/len(languages))
                count += 1
    return hypotheses, priors, languages

def generate_contexts(size):
    contexts = []
    if size == 3:
        for _ in range(25):
            for c in [[0.1, 0.2, 0.9], [0.1, 0.8, 0.9]]:
                contexts.append([c[0], c[1], c[2]])
                contexts.append([c[1], c[0], c[2]])
                contexts.append([c[1], c[2], c[0]])
                contexts.append([c[2], c[1], c[0]])
                contexts.append([c[2], c[0], c[1]])
                contexts.append([c[0], c[2], c[1]])
    elif size == 2:
        for _ in range(50):
            contexts.append([0.3, 0.1])
            contexts.append([0.4, 0.1])
            contexts.append([0.7, 0.9])
            contexts.append([0.6, 0.9])
            contexts.append([0.1, 0.3])
            contexts.append([0.1, 0.4])
            contexts.append([0.9, 0.7])
            contexts.append([0.9, 0.6])
            # # suggested contexts:
            # # 0.1, 0.7
            # # 0.3, 0.9
            # # 0.1, 0.6
            # # 0.4, 0.9
            # contexts.append([0.1, 0.7])
            # contexts.append([0.7, 0.1])
            # contexts.append([0.3, 0.9])
            # contexts.append([0.9, 0.3])
            # contexts.append([0.1, 0.6])
            # contexts.append([0.6, 0.1])
            # contexts.append([0.4, 0.9])
            # contexts.append([0.9, 0.4])
    else:
        print("Wrong context size")
        return None
    return np.array(contexts)