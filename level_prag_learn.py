import random
import numpy as np

from math import log, log1p, exp
from scipy.special import logsumexp, softmax

from copy import deepcopy

from argparse import ArgumentParser
import pickle

import utilities

""" generate list of language-perspective pairs and generate priors """
def generate_hypotheses():
    languages = utilities.generate_languages()
    hypotheses = []
    priors_unbiased = [log(1/len(perspectives)) + log(1/len(languages)) + log(1/len(pragmatic_levels)) for i in range(len(perspectives) * len(languages) * len(pragmatic_levels))]
    priors_egocentric = []
    for pl in pragmatic_levels:
        for p in perspectives:
            for l in languages:
                hypotheses.append([l, p, pl])

                p_prior = log(0.9) if p == p_learner else log(0.1)
                priors_egocentric.append(p_prior + log(1/len(languages)) + log(1/len(pragmatic_levels)))
    return [np.array(hypotheses), np.array(priors_unbiased), np.array(priors_egocentric)]

""" Pick a meaning with probability proportional to its designated probability """
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

""" The perspective-taking listener uses Bayes rule to compute the
    probability of a certain referent being intended by the speaker given the 
    produced signal, a language, and the listener's model of the speaker's 
    distribution over referents given their perspective """
def list1_lit_spkr(signal, meaning, language, ref_distribution):
    # get the list of signals which can be used for the given meaning
    signals_for_r = [signals[s] for s in range(len(meanings)) if language[meaning][s] == '1']
    num_signals_for_r = len(signals_for_r)
    # compute the product of the probability that the speaker chooses referent r and that signal s is produced 
    if signal in signals_for_r:
        if num_signals_for_r == num_signals:
            in_language = log(1 / num_signals_for_r)
        else:
            in_language = log((1 - noise) / num_signals_for_r)
        return ref_distribution[meaning] + in_language
    else:
        out_of_language = log(noise / (num_signals - num_signals_for_r))
        return ref_distribution[meaning] + out_of_language

def list1_perception_matrix(language, ref_distribution):
    mat = np.zeros((len(signals), len(meanings)))
    for s in range(len(signals)):
        row = np.zeros(len(meanings))
        for m in meanings:
            row[m] = list1_lit_spkr(signals[s], m, language, ref_distribution)
        mat[s] = utilities.normalize_logprobs(row)
    return mat

""" The level-2 pragmatic listener """
def list2_spkr1(signal, meaning, language, ref_distribution):
    # get the list of signals which can be used for the given meaning
    s_index = signals.index(signal)

    # compute the probability of the speaker producing the signal, with noise
    speaker_probs = spkr1_production_probs(meaning, language, ref_distribution)
    noisy_speaker_probs = deepcopy(speaker_probs)
    for s in range(len(noisy_speaker_probs)):
        noisy_speaker_probs[s] = noisy_speaker_probs[s] + log(1 - noise)
        other_signals = [noisy_speaker_probs[os] for os in range(len(noisy_speaker_probs)) if os != s]
        noisy_speaker_probs[s] = logsumexp([noisy_speaker_probs[s], (log(noise) + logsumexp(other_signals)) - log(len(signals) - 1)])

    return ref_distribution[meaning] + noisy_speaker_probs[s_index]

""" Update the posterior probabilities the learner has assigned to
    each lexicon/perspective pair based on the observed signal
    and context """
def update_posterior(posterior, signal, context):
    new_posterior = np.zeros(len(posterior))
    for i in range(len(posterior)): # for each hypothesis
        language = hypotheses[i][0]
        perspective = hypotheses[i][1]
        pragmatic_lvl = hypotheses[i][2]

        ref_distribution = calc_mental_state(perspective, context)

        marginalize = np.zeros(len(meanings))
        if pragmatic_lvl == 0:
            for m in meanings:
                marginalize[m] = list1_lit_spkr(signal, m, language, ref_distribution) # level-1 listener
        elif pragmatic_lvl == 1:
            for m in meanings:
                marginalize[m] = list2_spkr1(signal, m, language, ref_distribution) # level-2 listener
        
        new_posterior[i] = posterior[i] + logsumexp(marginalize)
    return utilities.normalize_logprobs(new_posterior)

""" The level-1 pragmatic speaker computes the probability of producing each signal
    given the intended meaning, their language, and their mental state
    (a distrubtion over meanings given their perspective and the context) """
def spkr1_production_probs(meaning, language, mental_state):
    # compute the utility of each signal as the negative surprisal of the intended
    # referent given the signal, for the listener
    signal_utility = np.array([alpha*list1_perception_matrix(language, mental_state)[s][meaning] for s in range(len(signals))])
    
    # use softmax to get distribution over signals
    return np.array([log(p) for p in softmax(signal_utility)])

""" Speaker produces a signal """
def produce(system, context):
    language = system[0]
    perspective = system[1]
    pragmatic_lvl = system[2]
    mental_state = calc_mental_state(perspective, context)
    meaning = sample(mental_state)
    
    """ Production is done differently depending on if the speaker is pragmatic or not """
    if pragmatic_lvl == 0:
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
    # choose the best signal given the pragmatically-derived probability distribution
    elif pragmatic_lvl == 1:
        signal = signals[utilities.log_roulette_wheel(spkr1_production_probs(meaning, language, mental_state))]
        
        # with small probability (noise), pick a different signal
        if random.random() < noise:
            other_signals = deepcopy(signals)
            other_signals.remove(signal)
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
    parser.add_argument("p", type=int, help="use pragmatic speakers", default="0")
    args = parser.parse_args()
    filename = args.o

    # non-pragmatic speakers
    if args.p == 0:
        s1 = 188 # lexicon where each meaning is associated with its corresponding signal
        s2 = 182 # lexicon where the last meaning is associated with all signals
        s3 = 171 # lexicon where only one signal is used for every meaning
    elif args.p == 1:
        # pragmatic speakers
        s1 = 874 # lexicon where each meaning is associated with its corresponding signal
        s2 = 868 # lexicon where the last meaning is associated with all signals
        s3 = 857 # lexicon where only one signal is used for every meaning

    # Generate maximally informative contexts, which are all possible permutations of
    # [0.1, 0.2, 0.9] and [0.1, 0.8, 0.9] (12 in total)
    contexts = []
    for _ in range(25):
        for c in [[0.1, 0.2, 0.9], [0.1, 0.8, 0.9]]:
            contexts.append([c[0], c[1], c[2]])
            contexts.append([c[1], c[0], c[2]])
            contexts.append([c[1], c[2], c[0]])
            contexts.append([c[2], c[1], c[0]])
            contexts.append([c[2], c[0], c[1]])
            contexts.append([c[0], c[2], c[1]])
    contexts = np.array(contexts)

    runs1 = []
    runs2 = []
    runs3 = []
    for _ in range(10):
        post_list1 = simulation(hypotheses[s1], 300, priors_egocentric, s1, contexts)
        runs1.append(post_list1)
        with open(filename + '_' + str(args.p) + '_runs1.pickle', 'wb') as f:
            pickle.dump(runs1, f)
        post_list2 = simulation(hypotheses[s2], 300, priors_egocentric, s2, contexts)
        runs2.append(post_list2)
        with open(filename + '_' + str(args.p) + '_runs2.pickle', 'wb') as f:
            pickle.dump(runs2, f)
        post_list3 = simulation(hypotheses[s3], 300, priors_egocentric, s3, contexts)
        runs3.append(post_list3)
        with open(filename + '_' + str(args.p) + '_runs3.pickle', 'wb') as f:
            pickle.dump(runs3, f)
    runs1 = np.array(runs1)
    runs2 = np.array(runs2)
    runs3 = np.array(runs3)
    data = np.array([runs1, runs2, runs3])
    with open(filename + '_' + str(args.p) + '_output.pickle', 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":  
    # Parameters
    noise = 0.05
    perspectives = [0, 1]
    pragmatic_levels = [0, 1]
    num_signals = 3
    meanings = [0, 1, 2]
    signals = ['a', 'b', 'c']
    p_learner = 1
    alpha = 3.0

    hypotheses, priors_unbiased, priors_egocentric = generate_hypotheses()
    main()