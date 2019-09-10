import pickle
import numpy as np

def get_sys_category(sys):
    # check what lexicon is being used
    if sys[1] == "0":
        cat = 0
    elif sys[1] in ["1", "2", "3", "6"]:
        cat = 1
    elif sys[1] in ["4", "8"]:
        cat = 2
    else:
        cat = 3

    if sys[5] == "1":
        # if this is a pragmatic system, add 4 to the category
        cat += 4
    return cat

def get_cat_name(cat):
    name = ""
    if cat < 4:
        name = "lit "
    else:
        name = "prag "
        cat = cat - 4
    if cat == 0:
        name += "filled"
    elif cat == 1:
        name += "ambig"
    elif cat == 2:
        name += "uninf"
    else:
        name += "perf inf"
    return name

matrix_sample = np.zeros((8,8))
matrix_map = np.zeros((8,8))
new_probs = {}
counts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0}
for spkr in range(36):
    with open('outputs_2x2/spkr' + str(spkr) + '_output_systems_2x2_sample_100runs_120bn.pickle', 'rb') as f:
        results = pickle.load(f)
        for input_sys,outputs in results.items():
            counts[get_sys_category(input_sys)] += 1
            for sys, prob in outputs.items():
                matrix_sample[get_sys_category(input_sys)][get_sys_category(sys)] += prob
row_sums = matrix_sample.sum(axis=1)
new_matrix = matrix_sample / row_sums[:, np.newaxis]
print(new_matrix)
w, v = np.linalg.eig(new_matrix.T)
w = w.real
v = v.real
u = (v[:,0]/v[:,0].sum()).real
print("Stationary distribution (with bottleneck 120)")
for i in range(8):
    print(get_cat_name(i), ":\t", u[i])
print("-----------")
for i in range(4):
    print(get_cat_name(i)[4:], ":\t", u[i]+u[i+4])
print("-----------")
for i in [0, 4]:
    print(get_cat_name(i)[:4], ":\t\t", np.sum(u[i:i+4]))