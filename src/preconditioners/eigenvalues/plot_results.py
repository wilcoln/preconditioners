import argparse
import os
import pickle
import numpy as np

import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True})

parser = argparse.ArgumentParser()
parser.add_argument('--folder', help='Folder of experiment', type=str)

args = parser.parse_args()

with open(os.path.join(args.folder, 'results.pkl'), 'rb') as f:
    results = pickle.load(f)

widths = []
min_evalues = []
max_evalues = []
avg_evalues = []
all_evalues = []

for entry in results:
    widths.append(entry['width'])
    evalues = entry['eigenvalues']
    min_evalues.append(np.min(evalues))
    max_evalues.append(np.max(evalues))
    avg_evalues.append(np.average(evalues))
    all_evalues.append(evalues)

# Plot min and max eigenavlues
plt.scatter(widths, min_evalues, label='Min Eigenvalue')
plt.scatter(widths, avg_evalues, label='Average Eigenvalue')
plt.scatter(widths, max_evalues, label='Max Eigenvalue')
plt.legend()
plt.ylabel(r'$\lambda$')
plt.xlabel(r'$n$')
plt.show()

# Plot all eigenvalues
plt.scatter(widths * len(all_evalues[0]), np.array(all_evalues).transpose().flatten())
plt.ylabel(r'$\lambda$')
plt.xlabel(r'$n$')
plt.show()
