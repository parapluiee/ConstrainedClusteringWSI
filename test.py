import numpy as np

arr = np.array(range(1,100))

print(arr)

seeds = [[0], [2, 39, 18], [77, 1]]
seed_dict = {s:i for i, seed in enumerate(seeds) for s in seed}

print(seed_dict)
all_seeds = [s for row in seeds for s in row]
all_seeds.sort()
seed_assignment = [seed_dict[s] for s in all_seeds]
mask = np.ones(len(arr), dtype=bool)

for s in all_seeds:
    mask[s] = False

print(arr)
arr[np.invert(mask)] = seed_assignment
print(arr)
