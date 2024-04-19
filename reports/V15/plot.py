import matplotlib.pyplot as plt
import numpy as np

U, I = np.genfromtxt("data/Leak_current/leakage_current.txt", unpack = True)

plt.scatter(U, I, label='Leakage current', marker = "x")
plt.vlines(x = 60, ymin= 0.75, ymax=1.9, linestyles="dashed", colors="black", label = "Depletion voltage")
plt.xlabel(r'$U \mathbin{/} \unit{\volt}$')
plt.ylabel(r'$I \mathbin{/} \unit{\micro\ampere}$')
plt.grid()
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/leakage.pdf')

ped = np.genfromtxt("data/Pedestal/Pedestal.txt", unpack = True, delimiter=";", dtype=float)
