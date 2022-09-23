import os
import numpy as np
from src import project_dir
import matplotlib.pyplot as plt

random_no_do = np.loadtxt(os.path.join(project_dir, 'results/results_1.txt'))
random = np.loadtxt(os.path.join(project_dir, 'results/results_2.txt'))
max_entropy = np.loadtxt(os.path.join(project_dir, 'results/results_3.txt'))
bald = np.loadtxt(os.path.join(project_dir, 'results/results_4.txt'))

fig, ax = plt.subplots(figsize=(8,6))

ax.set_ylim([0.8, 1])

ax.plot(random_no_do[:,0], random_no_do[:,1], label='Random (without dropout)')
ax.plot(random[:,0], random[:,1], label='Random')
ax.plot(max_entropy[:,0], max_entropy[:,1], label='Max entropy')
ax.plot(bald[:,0], bald[:,1], label='BALD')
ax.legend()
ax.set_xlabel('Number of samples')
ax.set_ylabel('Test accuracy')
plt.grid(True)

fig.savefig(os.path.join(project_dir, 'results/al_curve.pdf'))
