import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from uncertainties import ufloat

## 1) Depletion voltage ###

U, I = np.genfromtxt("data/Leak_current/leakage_current.txt", unpack = True)

plt.scatter(U, I, label='Leakage current', marker = "x")
plt.vlines(x = 60, ymin= 0.75, ymax=1.9, linestyles="dashed", colors="black", label = "Depletion voltage")
plt.xlabel(r'$U \mathbin{/} \unit{\volt}$')
plt.ylabel(r'$I \mathbin{/} \unit{\micro\ampere}$')
plt.grid()
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/leakage.pdf')
plt.close()

## 2) Pedestal run ##

ADC = np.genfromtxt("data/Pedestal/Pedestal.txt", unpack = True, delimiter=";", dtype=float)

pedestal = np.mean(ADC, axis=0)
common_mode = np.mean(ADC-pedestal, axis=1)
noise = np.sqrt( 1/(len(ADC)-1) * np.sum((((ADC - pedestal).T - common_mode).T)**2, axis=0))

plt.bar(np.linspace(1,128,128), pedestal, width=1)
plt.ylim(500,520)
plt.xlabel(r'Strip')
plt.ylabel(r'Pedestal')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/pedestal.pdf')
plt.close()

plt.bar(np.linspace(1,128,128), noise, width=1)
plt.ylim(1.7,2.4)
plt.xlabel(r'Strip')
plt.ylabel(r'Noise')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/noise.pdf')
plt.close()

plt.hist(common_mode, bins=20, density=True)
plt.xlabel(r'Common Mode Shift / ADC counts')
plt.ylabel(r'Probability')
plt.savefig('build/common_mode.pdf')
plt.close()

## 3) Calibration measurements ##

t, ADC = np.genfromtxt("data/Calib/delay.txt", unpack = True)

plt.scatter(t, ADC, label = "Meausured counts")
plt.scatter(t[8], ADC[8],marker="x", color = "red", label = "Otimal delay value")
plt.xlabel(r'$t$ / nS')
plt.ylabel(r'ADC counts')
plt.grid()
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/delay.pdf')
plt.close()

charge_20, ADC_20 = np.genfromtxt("data/Calib/charge_20.txt", unpack = True)
charge_40, ADC_40 = np.genfromtxt("data/Calib/charge_40.txt", unpack = True)
charge_60, ADC_60 = np.genfromtxt("data/Calib/charge_60.txt", unpack = True)
charge_80, ADC_80 = np.genfromtxt("data/Calib/charge_80.txt", unpack = True)
charge_100, ADC_100 = np.genfromtxt("data/Calib/charge_100.txt", unpack = True)
charge_60_0V, ADC_60_0V = np.genfromtxt("data/Calib/charge_60_0volts.txt", unpack = True)

ADC_mean = (ADC_20+ADC_40+ADC_60+ADC_80+ADC_100)/5

plt.scatter(charge_20, ADC_20, label = "Channel 20", marker="x")
plt.scatter(charge_40, ADC_40, label = "Channel 40", marker="x")
plt.scatter(charge_60, ADC_60, label = "Channel 60", marker="x")
plt.scatter(charge_80, ADC_80, label = "Channel 80", marker="x")
plt.scatter(charge_100, ADC_100, label = "Channel 100", marker="x")
plt.plot(charge_20, ADC_mean, color = "red", label = "Mean value")
plt.grid()
plt.xlabel(r'Injected charge / e')
plt.ylabel(r'ADC counts')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/calib.pdf')
plt.close()

plt.scatter(charge_60, ADC_60, label = "Channel 60", marker="x")
plt.plot(charge_60_0V, ADC_60_0V, label = "Channel 60 at 0V", color = "red")
plt.grid()
plt.xlabel(r'Injected charge / e')
plt.ylabel(r'ADC counts')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.xlim(0.5*10**5,1.5*10**5)
plt.savefig('build/calib_0V.pdf')
plt.close()


def conversion(x, a, b, c, d, e):
    return (a*x**4 + b*x**3 + c*x**2 + d*x + e)
cutoff = 250
fit_range = ADC_mean<cutoff
params, cov = curve_fit(conversion, ADC_mean[fit_range], charge_20[fit_range], p0=[0, 0, 1, 10, 1000])
errors = np.sqrt(np.diag(cov))
a = ufloat(params[0],errors[0])
b = ufloat(params[1],errors[1])
c = ufloat(params[2],errors[2])
d = ufloat(params[3],errors[3])
e = ufloat(params[4],errors[4])
print("Fit parameters for conversion:")
print(f"a={a}")
print(f"b={b}")
print(f"c={c}")
print(f"d={d}")
print(f"e={e}")
print("")

plt.scatter(ADC_mean, charge_20,  label = "Mean ADC counts", marker="x")
plt.plot(ADC_mean, conversion(ADC_mean, *params), label = "Fit", color = "red")
plt.vlines(cutoff, ymin=0, ymax=2.5*10**5, label = "End of fit range", color = "black", linestyle = "dashed") 
plt.grid()
plt.xlabel(r'Mean ADC counts')
plt.ylabel(r'Injected charge / e')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/calib_fit.pdf')
plt.close()

## 4) Laser measurement ##

t, ADC = np.genfromtxt("data/Laser_sync/lasersync(delay100).txt", unpack = True)

plt.scatter(t, ADC, label = "ADC counts")
plt.vlines(100, ymin=0, ymax=145, label = "Optimal delay at 100 nS", color = "red", linestyle = "dashed")
plt.xlabel(r'Delay / nS')
plt.ylabel(r'ADC counts')
plt.grid()
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/laser_delay.pdf')
plt.close()

ADC = np.genfromtxt("data/Laser_scan/Laserscan.txt", unpack = True, dtype=float)
x,y = np.meshgrid(np.linspace(0,34,35),np.linspace(1,128,128))
plt.pcolormesh(x, y, ADC, cmap="inferno")
plt.xlabel(r'Distance / $\mu$m')
plt.ylabel(r'Channel')
plt.ylim(80,87)
plt.savefig('build/laser_scan.pdf')
plt.close()

plt.plot(np.linspace(0,34,35), ADC[82,:], label="Measured counts")
plt.xlim(8,35)
plt.vlines(11, ymin=0, ymax=140, label = "Start of Peak", color = "red", linestyle = "dashed")
plt.vlines(23, ymin=0, ymax=140, label = "End of Peak", color = "red", linestyle = "dashdot")
plt.vlines(17, ymin=0, ymax=140, label = "Maximum", color = "green", linestyle = "dashed") 
plt.xlabel(r'Distance / $\mu$m')
plt.ylabel(r'ADC counts')
plt.grid()
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/channel_83.pdf')
plt.close()

## 5) Charge collection efficiency ##

i=0
idx = 0
ccel = np.zeros(shape=(21,128))
while(i<210):
    ccel[idx,:] = np.genfromtxt(f"data/CCEL/{i}CCEL.txt", unpack = True)
    i+=10
    idx+=1

x,y = np.meshgrid(np.linspace(0,200,21),np.linspace(1,128,128))
plt.pcolormesh(x, y, ccel.T, cmap="inferno")
plt.ylim(68,76)
plt.xlabel(r'$U \mathbin{/} \unit{\volt}$')
plt.ylabel(r'Channel')
plt.savefig('build/CCEL_channel.pdf')
plt.close()

norm = np.copy(ccel[8,:])
for j in range(21):
    ccel[j,:] = ccel[j,:]/norm

D = 300*10**-6
U_dep = 80
def CCE(U, a):
    return (1 - np.exp(- (D*np.sqrt(U/U_dep)) / a)) / (1 - np.exp(- D / a))
params_, cov = curve_fit(CCE, np.linspace(0,80,9) ,ccel[0:9,71], bounds=((0),(300*10**-6)))
a = ufloat(params_[0],errors[0])
#U = ufloat(params[1],errors[1])
print("Fit parameters for CCE with laser:")
print(f"a={a*10**6} micrometers")
#print(f"U_dep={U} volts")
print("")

U = np.linspace(0,200,21)
plt.scatter(U, ccel[:,71], label='Measurement', marker = "x")
plt.vlines(80, ymin=0, ymax=1.1, label = "Start of Plateu at U_dep", color = "black", linestyle = "dashed")
plt.plot(np.linspace(0,80,100), CCE(np.linspace(0,80,100), *params_), label = "Fit", color = "red")
plt.xlabel(r'$U \mathbin{/} \unit{\volt}$')
plt.ylabel(r'CCE / %')
plt.grid()
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/CCEL.pdf')
plt.close()

## 6) Charge collection with source ##

mean_counts = []
U = np.arange(0, 201, 10)
clusters = []
for i in range(128):
    clusters.append(f"{i}")
for volt in U:
    cceq =  pd.read_csv(f'data/CCEQ/{volt}_Cluster_adc_entries.txt', sep = '\t', names = clusters)
    mean_counts.append(cceq.sum(axis=1).mean())
mean_counts = mean_counts / mean_counts[9]

#params, cov = curve_fit(CCE, np.linspace(0,80,9) ,mean_counts[:9], bounds=((0),(300*10**-6)))
#a = ufloat(params[0],errors[0])
#U = ufloat(params[1],errors[1])
#print("Fit parameters for CCE with source:")
#print(f"a={a*10**6} micrometers")
#print(f"U_dep={U} volts")
#print("")

plt.plot(U, mean_counts, label='Measurement', marker = "x")
plt.vlines(80, ymin=0, ymax=1.1, label = "Start of Plateu at U_dep", color = "black", linestyle = "dashed")
#plt.plot(np.linspace(0,80,100), CCE(np.linspace(0,80,100), *params), label = "Fit", color = "red")
plt.xlabel(r'$U \mathbin{/} \unit{\volt}$')
plt.ylabel(r'Mean CCE / %')
plt.grid()
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/CCEQ.pdf')
plt.close()

## 7) Large source scan ##

no_clusters = np.genfromtxt("data/RS_scan/number_of_clusters.txt", unpack = True)
plt.bar(np.linspace(1,128,128), no_clusters)
plt.yscale("log")
plt.xlim(0,10)
plt.xlabel(r'Number of clusters')
plt.ylabel(r'Counts')
plt.grid(axis="y")
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/num_clusters.pdf')
plt.close()

no_channels = np.genfromtxt("data/RS_scan/cluster_size.txt", unpack = True)
plt.bar(np.linspace(1,128,128), no_channels)
plt.yscale("log")
plt.xlim(0,18)
plt.xlabel(r'Number of channels per cluster')
plt.ylabel(r'Counts')
plt.grid(axis="y")
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/num_channels.pdf')
plt.close()

hits = np.genfromtxt("data/RS_scan/hitmap.txt", unpack = True)
plt.bar(np.linspace(1,128,128), hits)
plt.xlim(0,128)
plt.xlabel(r'Channel')
plt.ylabel(r'Hits')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/hitmap.pdf')
plt.close()

entries = np.genfromtxt("data/RS_scan/cluster_ADCs.txt", unpack = True)
plt.hist(entries, bins = np.arange(1,300,1))
plt.xlabel(r'ADC counts')
plt.ylabel(r'Number of events')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/ADC_counts.pdf')
plt.close()

energies = conversion(entries, *params)
energies = energies*10**-3
cut = energies<400
plt.hist(energies[cut], bins = 80, histtype="step", label="Measured energy distribution")
plt.vlines(26, ymin=0, ymax=21*10**4, label = "MPV", color = "black", linestyle = "dashed")
plt.vlines(np.mean(energies), ymin=0, ymax=21*10**4, label = "Mean", color = "red", linestyle = "dashed")
plt.xlim(0,250)
plt.xlabel(r'Energy / keV')
plt.ylabel(r'Counts per 10 keV')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/energy.pdf')
plt.close()