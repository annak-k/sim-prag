import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from argparse import ArgumentParser
import pickle

def plot_graph(name, title, results_list):
    colors = ['darkseagreen', 'steelblue', 'mediumpurple']
    labels = ['Most informative', 'Medium informative', 'Least informative']
    
    for i in range(len(results_list)):
        average = np.zeros(len(results_list[i][0]))
        for result in results_list[i]:
            plt.plot(result, color=colors[i], alpha=0.3)

        for j in range(len(results_list[i][0])):
            total = 0
            for result in results_list[i]:
                total += result[j]
            average[j] = total / len(results_list[i])

        plt.plot(average, color=colors[i], label=labels[i])
    plt.xlabel('utterances observed by the learner')
    plt.ylabel('learner\'s posterior')
    plt.legend()
    plt.grid()
    plt.title(title)
    plt.savefig(name + '_plot.png')

def main():
    parser = ArgumentParser()
    parser.add_argument("f", type=str, help="the name of the file containing the results to plot")
    parser.add_argument("t", type=str, help="title of the plot", default="")
    args = parser.parse_args()
    
    with open(args.f + '.pickle', 'rb') as f:
        results = pickle.load(f)
        plot_graph(args.f, args.t, results)

if __name__ == "__main__":
    main()