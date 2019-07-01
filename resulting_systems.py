import pickle

matrix = {}
for spkr in ["171", "182", "188", "857", "868", "874"]:
    with open('spkr' + spkr + '_output_systems_sample.pickle', 'rb') as f:
        matrix = {**matrix, **pickle.load(f)}
# print(matrix)

print("Input system  --> Output systems")
for k, v in matrix.items():
    print(k, end=" --> ")
    for h, p in v.items():
        print(h + ": " + '{:.3}'.format(p), end="  ")
    print("")