"""Plots results from the training_comparison experiment"""
import argparse
import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True})

parser = argparse.ArgumentParser()
parser.add_argument('--folder', help='Folder of model logs', type=str)

args = parser.parse_args()

test_errors = defaultdict(list)

for experiment_file in os.listdir(args.folder):
    if experiment_file[:-4] != ".pkl":
        continue
    with open(os.path.join(args.folder, experiment_file), 'rb') as f:
        results = pickle.load(f)

    s2 = results['sigma2']
    test_loss = results['test_loss']

    print(f"Analysing sigma2={s2}")
    print(experiment_folder)

    test_errors[s2].append(test_loss)

var_vals = sorted(list(var_to_loss.keys()))
average_test_loss = [np.average(var_vals[s2]) for s2 in test_errors]

# Plot test loss against variance
plt.scatter(var_vals, test_loss, label="KFAC")
plt.scatter(var_vals, test_loss_extra, label="KFAC-extra")
plt.ylabel("Test Loss")
plt.xlabel(r'$\sigma^2$')
if args.line_of_best_fit:
    plt.gca().set_prop_cycle(None)
    x = np.linspace(var_vals[0], var_vals[-1], 10)
    best_fit = np.poly1d(np.polyfit(var_vals, test_loss, 1))(x)
    best_fit_extra = np.poly1d(np.polyfit(var_vals, test_loss_extra, 1))(x)
    plt.plot(x, best_fit)
    plt.plot(x, best_fit_extra)
plt.legend()
plt.show()



# Plot the mean test errors
for optim_cls in OPTIMIZER_CLASSES:
    plt.scatter(noise_variances, mean_test_errors[optim_cls.__name__], label=optim_cls.__name__)

    plt.xlabel('Noise variance')
    plt.ylabel('Test loss')
    plt.legend([optim_cls.__name__ for optim_cls in OPTIMIZER_CLASSES])
