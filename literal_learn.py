import random
import numpy as np

from math import log, log1p, exp
from scipy.special import logsumexp

from copy import deepcopy

from argparse import ArgumentParser
import pickle

import utilities

""" generate list of language-perspective pairs and generate priors """
def generate_hypotheses():
    languages = utilities.generate_languages()
    lp_pairs = []
    priors_unbiased = [log(1/len(perspectives)) + log(1/len(languages)) for i in range(len(perspectives) * len(languages))]
    priors_egocentric = []
    for p in perspectives:
        for l in languages:
            lp_pairs.append([l, p])

            p_prior = log(0.9) if p == p_learner else log(0.1)
            priors_egocentric.append(p_prior + log(1/len(languages)))
    return [np.array(lp_pairs), np.array(priors_unbiased), np.array(priors_egocentric)]

def sample(posterior):
    return meanings[utilities.log_roulette_wheel(posterior)]

""" Given speaker's perspective and the context,
    compute a probability distribution over the referents
    of how likely the speaker is to speak about each referent
    p. 88 Equation 3.1 """
def calc_mental_state(perspective, context):
    distribution = np.zeros(len(context))
    for o in range(len(context)):
        distribution[o] = log(1 - abs(perspective - context[o]))
    return utilities.normalize_logprobs(distribution)

""" Update the posterior probabilities the learner has assigned to
    each lexicon/perspective pair based on the observed signal
    and context """
def update_posterior(posterior, signal, context):
    new_posterior = np.zeros(len(posterior))
    for i in range(len(posterior)): # for each hypothesis
        language = lp_pairs[i][0]
        perspective = lp_pairs[i][1]

        ref_distribution = calc_mental_state(perspective, context)

        marginalize = np.zeros(len(meanings))
        for m in meanings:
            # get the list of signals which can be used for the given meaning
            signals_for_r = [signals[s] for s in range(len(meanings)) if language[m][s] == '1']
            num_signals_for_r = len(signals_for_r)
            # compute the product of the probability that the speaker chooses referent r and that signal s is produced 
            if signal in signals_for_r:
                if num_signals_for_r == num_signals:
                    in_language = log(1 / num_signals_for_r)
                else:
                    in_language = log((1 - noise) / num_signals_for_r)
                marginalize[m] = ref_distribution[m] + in_language
            else:
                out_of_language = log(noise / (num_signals - num_signals_for_r))
                marginalize[m] = ref_distribution[m] + out_of_language
        
        new_posterior[i] = posterior[i] + logsumexp(marginalize)
    return utilities.normalize_logprobs(new_posterior)

def produce(system, context):
    language = system[0]
    meaning = sample(calc_mental_state(system[1], context))
    signal = signals[utilities.wta(language[meaning])]

    signals_for_r = [signals[s] for s in range(len(meanings)) if language[meaning][s] == '1']
    num_signals_for_r = len(signals_for_r)
    # with small probability (noise), pick a signal that doesn't correspond to
    # the selected meaning in the given language
    if random.random() < noise and num_signals_for_r != 3:
        other_signals = deepcopy(signals)
        for s in signals_for_r:
            other_signals.remove(s)
        signal = random.choice(other_signals) 
    return [signal, context]

def simulation(speaker, no_productions, priors, hypoth_index, contexts):
    posteriors = deepcopy(priors)
    posterior_list = [exp(posteriors[hypoth_index])]
    for i in range(no_productions):
        d = produce(speaker, contexts[i])
        posteriors = update_posterior(posteriors, d[0], d[1])
        posterior_list.append(exp(posteriors[hypoth_index]))
    return np.array(posterior_list)

def main():
    parser = ArgumentParser()
    parser.add_argument("o", type=str, help="prefix for the output files")
    args = parser.parse_args()
    filename = args.o
    
    speaker1 = lp_pairs[188] # lexicon where each meaning is associated with its corresponding signal
    speaker2 = lp_pairs[182] # lexicon where the last meaning is associated with all signals
    speaker3 = lp_pairs[171] # lexicon where only one signal is used for every meaning
    contexts = np.array([[random.random(), random.random(), random.random()] for i in range(500)])

    runs1 = []
    runs2 = []
    runs3 = []

    for _ in range(10):
        post_list1 = simulation(speaker1, 300, priors_unbiased, 188, contexts)
        runs1.append(post_list1)
        with open(filename + '_runs1.pickle', 'wb') as f:
            pickle.dump(runs1, f)
        post_list2 = simulation(speaker2, 300, priors_unbiased, 182, contexts)
        runs2.append(post_list2)
        with open(filename + '_runs2.pickle', 'wb') as f:
            pickle.dump(runs2, f)
        post_list3 = simulation(speaker3, 300, priors_unbiased, 171, contexts)
        runs3.append(post_list3)
        with open(filename + '_runs3.pickle', 'wb') as f:
            pickle.dump(runs3, f)
    runs1 = np.array(runs1)
    runs2 = np.array(runs2)
    runs3 = np.array(runs3)
    data = np.array([runs1, runs2, runs3])
    with open(filename + '_output.pickle', 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":  
    # Parameters
    noise = 0.05
    perspectives = [0, 1]
    num_signals = 3
    meanings = [0, 1, 2]
    signals = ['a', 'b', 'c']
    p_learner = 1

    lp_pairs, priors_unbiased, priors_egocentric = generate_hypotheses()
    main()