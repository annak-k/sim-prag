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
    with open('outputs_2x2/spkr' + str(spkr) + '_output_systems_2x2_sample.pickle', 'rb') as f:
        results = pickle.load(f)
        for input_sys,outputs in results.items():
            
            # if get_sys_category(input_sys) not in new_probs:
            #     new_probs[get_sys_category(input_sys)] = {}
            #     counts[get_sys_category(input_sys)] = 1
            # else:
            counts[get_sys_category(input_sys)] += 1
            for sys, prob in outputs.items():
                matrix_sample[get_sys_category(input_sys)][get_sys_category(sys)] += prob
            #     if get_sys_category(sys) not in new_probs[get_sys_category(input_sys)]:
            #         new_probs[get_sys_category(input_sys)][get_sys_category(sys)] = prob
            #     else:
            #         new_probs[get_sys_category(input_sys)][get_sys_category(sys)] += prob
            #     # print(get_sys_category(sys) + ": " + str(prob), end=", ")
row_sums = matrix_sample.sum(axis=1)
new_matrix = matrix_sample / row_sums[:, np.newaxis]
# print(new_matrix.T)
w, v = np.linalg.eig(new_matrix.T)
w = w.real
v = v.real
u = (v[:,0]/v[:,0].sum()).real
print("Stationary distribution (with bottleneck 60)")
for i in range(8):
    print(get_cat_name(i), ":\t", u[i])
print("-----------")
for i in range(4):
    print(get_cat_name(i)[4:], ":\t", u[i]+u[i+4])
print("-----------")
for i in [0, 4]:
    print(get_cat_name(i)[:4], ":\t\t", np.sum(u[i:i+4]))
# print(np.dot(v[:,0].T, new_matrix).T)
# for r in range(len(new_matrix)):
#     print(get_cat_name(r), ": ", [get_cat_name(s) + str(new_matrix[r][s]) for s in np.where(new_matrix[r] != 0)[0]])
# for s, o in new_probs.items():
#     count = counts[s]
#     for sys, p in o.items():
#         new_probs[s][sys] = p / count
# print(new_probs)
        # matrix_sample = {**matrix_sample, **pickle.load(f)}
#     with open('spkr' + str(spkr) + '_output_systems_2x2_map.pickle', 'rb') as f:
#         matrix_map = {**matrix_map, **pickle.load(f)}
# print(matrix)

# print("SAMPLE")
# print("Input system  --> Output systems")
# for k, v in matrix_sample.items():
#     print(k, end=" --> ")
#     for h, p in v.items():
#         print(h + ": " + '{:.3}'.format(p), end="  ")
#     print("")

# print("MAP")
# print("Input system  --> Output systems")
# for k, v in matrix_map.items():
#     print(k, end=" --> ")
#     for h, p in v.items():
#         print(h + ": " + '{:.3}'.format(p), end="  ")
#     print("")