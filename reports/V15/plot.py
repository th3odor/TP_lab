import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat

## 1) Depletion voltage ###

U, I = np.genfromtxt("data/Leak_current/leakage_current.txt", unpack = True)

plt.scatter(U, I, label='Leakage current', marker = "x")
plt.vlines(x = 60, ymin= 0.75, ymax=1.9, linestyles="dashed", colors="black", label = "Depletion voltage")
#plt.xlabel(r'$U \mathbin{/} \unit{\volt}$')
#plt.ylabel(r'$I \mathbin{/} \unit{\micro\ampere}$')
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
plt.xlabel(r'Common Mode Shift')
plt.ylabel(r'Probability')
plt.savefig('build/common_mode.pdf')
plt.close()

## 3) Calibration measurements ##

t, ADC = np.genfromtxt("data/Calib/delay.txt", unpack = True)

plt.scatter(t, ADC, label = "Meausured counts")
plt.scatter(t[8], ADC[8],marker="x", color = "red", label = "Otimal delay value")
plt.xlabel(r'Time [nS]')
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
plt.xlabel(r'Injected charge')
plt.ylabel(r'ADC counts')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/calib.pdf')
plt.close()

plt.scatter(charge_60, ADC_60, label = "Channel 60", marker="x")
plt.plot(charge_60_0V, ADC_60_0V, label = "Channel 60 at 0V", color = "red")
plt.grid()
plt.xlabel(r'Injected charge')
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

plt.scatter(ADC_mean, charge_20,  label = "Mean ADC counts", marker="x")
plt.plot(ADC_mean, conversion(ADC_mean, *params), label = "Fit", color = "red")
plt.vlines(cutoff, ymin=0, ymax=2.5*10**5, label = "End of fit range", color = "black", linestyle = "dashed") 
plt.grid()
plt.xlabel(r'Mean ADC counts')
plt.ylabel(r'Injected charge')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/calib_fit.pdf')
plt.close()

## 4) Laser measurement ##

t, ADC = np.genfromtxt("data/Laser_sync/lasersync(delay100).txt", unpack = True)

plt.scatter(t, ADC, label = "ADC counts")
plt.vlines(100, ymin=0, ymax=145, label = "Optimal delay at 100 nS", color = "red", linestyle = "dashed")
plt.xlabel(r'Delay [nS]')
plt.ylabel(r'ADC counts')
plt.grid()
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/laser_delay.pdf')
plt.close()

ADC = np.genfromtxt("data/Laser_scan/Laserscan.txt", unpack = True, dtype=float)
x,y = np.meshgrid(np.linspace(0,34,35),np.linspace(1,128,128))
plt.pcolormesh(x, y, ADC, cmap="inferno")
plt.xlabel(r'Distance in um')
plt.ylabel(r'Channel')
plt.ylim(65,95)
plt.savefig('build/laser_scan.pdf')
plt.close()
##fuckkkkk