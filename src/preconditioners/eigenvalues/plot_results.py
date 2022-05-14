"""Plots results from the kernel_eigenvalues and fisher_eigenvalues experiment"""
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
    evalues = [np.real(e) for e in entry['eigenvalues']]
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

# Plot min eigenavlues
plt.scatter(widths, min_evalues)
plt.ylabel(r'$\lambda_{min}$')
plt.xlabel(r'$n$')
plt.yscale("log")
plt.show()

print(f"Maximum number of parameters: {len(all_evalues[-1])}")
print(min_evalues)

# Plot all eigenvalues
for i in range(len(widths)):
    plt.scatter([float(widths[i])]*len(all_evalues[i]), all_evalues[i], c='#1f77b4')
plt.ylabel(r'$\lambda$')
plt.xlabel(r'$n$')
plt.show()