import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
import seaborn as sns
from scipy.optimize import curve_fit
from copy import deepcopy


angleC = []
angleDC = []
for i in range(45):
    angleC.append(pd.read_csv(f'data_local/TheoCanAngle/Attenuation_h={i}deg_v=0deg_x=0mm.txt', sep='\t'))
    angleC.append(pd.read_csv(f'data_local/TheoCanAngle/Attenuation_h={i}.5deg_v=0deg_x=0mm.txt', sep='\t'))
    angleDC.append(pd.read_csv(f'data_local/TheoCanAngle/DarkCounts_h={i}deg_v=0deg.txt', sep='\t'))
    angleDC.append(pd.read_csv(f'data_local/TheoCanAngle/DarkCounts_h={i}.5deg_v=0deg.txt', sep='\t'))



angleInt = np.zeros(90)
for i in range(90):
    angleInt[i] = (angleC[i] - angleDC[i])['C1'].sum()
angle = np.linspace(0,44.5,90)

print(angleInt*np.sin(angle/90*np.pi))

counts, bins, stff = plt.hist(angle, weights=angleInt*np.sin(angle/90*np.pi), bins=45,range=(0,45), histtype='step', linewidth = 2,color='tab:orange')
KYS = angleInt*np.sin(angle/90*np.pi)
plt.axvline(x=bins[int(KYS.argmax()*0.5)    ]+(bins[1]-bins[0])/2, label=r'$\theta_\mathrm{max}$', linewidth=2, c='tab:red')

plt.xlabel("Horizontal Angle / Degrees")
plt.ylabel("Counts / a.u.")
plt.legend();
plt.tight_layout()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig('build/MaxAngle.pdf')
plt.close()


print('Max angle:',bins[int(KYS.argmax()*0.5)]+(bins[1]-bins[0])/2)


sim=[]
N = 1200

for i in range(N):
    sim.append(pd.read_csv(f'data_local/sim2/job_{i}.txt', sep="\t",encoding='iso-8859-1', on_bad_lines='skip'))
    

lichtAn = pd.read_csv('data_local/TheoCanLightDark_lon.txt', sep='\t')
lichtAus = pd.read_csv('data_local/TheoCanLightDark_loff.txt', sep='\t')


plt.scatter(lichtAn[lichtAn.columns[0]],lichtAn[lichtAn.columns[2]],s=20,marker='x',linewidths=1,label='Light On')
plt.scatter(lichtAus[lichtAus.columns[0]],lichtAus[lichtAus.columns[2]],s=20,marker='+',linewidths=1,label='Light Off')
plt.legend()
plt.xlabel(r'$\lambda/$nm')
plt.ylabel('Counts / a.u');
plt.tight_layout()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.savefig('build/LightOnOff.pdf')
plt.close()

plt.scatter(lichtAn[lichtAn.columns[0]],lichtAn[lichtAn.columns[2]]-lichtAn[lichtAn.columns[1]],s=20,marker='x',linewidths=1,label='Light On')
plt.scatter(lichtAus[lichtAus.columns[0]],lichtAus[lichtAus.columns[2]]-lichtAus[lichtAus.columns[1]],s=20,marker='+',linewidths=1,label='Light Off')
plt.legend()
plt.xlabel(r'$\lambda/$nm')
plt.ylabel('Counts / a.u');
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.tight_layout()
plt.savefig('build/DCsubtracted.pdf')
plt.close()

radialC = np.empty((15,15))
radialDC = np.empty((15,15))
for (i,h) in enumerate(np.linspace(-18,26.8,15)):
    for (j,v) in enumerate(np.linspace(-6,31.8,15)):
        # print(i,j,"   ",h,v.round(1))
        try:
            radC_ = pd.read_csv(f"data_local/TheoCanRadial/Attenuation_h={h:.1f}deg_v={v:.1f}deg_x=0mm.txt",sep="\t")
            radDC_ = pd.read_csv(f"data_local/TheoCanRadial/DarkCounts_h={h:.1f}deg_v={v:.1f}deg.txt",sep="\t")
        except:
            try:
                radC_ = pd.read_csv(f"data_local/TheoCanRadial/Attenuation_h={h:.0f}deg_v={v:.1f}deg_x=0mm.txt",sep="\t")
                radDC_ = pd.read_csv(f"data_local/TheoCanRadial/DarkCounts_h={h:.0f}deg_v={v:.1f}deg.txt",sep="\t")
            except:
                try:
                    radC_ = pd.read_csv(f"data_local/TheoCanRadial/Attenuation_h={h:.1f}deg_v={v:.0f}deg_x=0mm.txt",sep="\t")
                    radDC_ = pd.read_csv(f"data_local/TheoCanRadial/DarkCounts_h={h:.1f}deg_v={v:.0f}deg.txt",sep="\t")
                except:
                    radC_ = pd.read_csv(f"data_local/TheoCanRadial/Attenuation_h={h:.0f}deg_v={v:.0f}deg_x=0mm.txt",sep="\t")
                    radDC_ = pd.read_csv(f"data_local/TheoCanRadial/DarkCounts_h={h:.0f}deg_v={v:.0f}deg.txt",sep="\t")
        radialC[i,j] =  radC_[radC_.columns[-1]].sum()
        radialDC[i,j] =  radDC_[radDC_.columns[-1]].sum()

rad = (radialC - radialDC)/np.max((radialC - radialDC))


sns.heatmap(rad, yticklabels=np.linspace(-18,26.8,15).round(1), xticklabels=np.linspace(-6,31.8,15).round(1), cmap='Reds')
plt.xlabel('Vertical angle / Degrees')
plt.ylabel('Horizontal angle / Degrees');
plt.tight_layout()
plt.savefig('build/angles.pdf')
plt.close()

col = sim[0].columns


sim_ = sim[0]
for i in range(N):
    sim[i]['r_exit'] = np.sqrt(sim[i][col[0]]**2 + sim[i][col[1]]**2)
    sim_['r_exit'] = np.sqrt(sim_[col[0]]**2 + sim_[col[1]]**2)
    sim[i] = sim[i][((sim[i]['r_exit']<0.125) & (sim[i]['rayleighScatterings']==0.0))]



counts, bins, stff = plt.hist(sim_['r_exit'],histtype='step', bins=50, linewidth=1.5, label='before',range=(0, 0.6))
plt.hist(sim[0]['r_exit'],histtype='step', bins=bins, linewidth=1.5, label='After');
plt.axvline(x=0.125, label='Cut', c='tab:red')
plt.xlabel(r"$r_{\mathrm{start}}$ / mm")
plt.ylabel("Counts / a.u.")
plt.legend();
plt.tight_layout()
plt.savefig('build/rCut.pdf')
plt.close()

simKern = []
simMantel = []

for i in range(N):
    simKern.append(deepcopy(sim[i][(sim[i]['length_clad']==0.0)]))
    simMantel.append(deepcopy(sim[i][(sim[i]['length_clad']!=0.0)]))

for i in range(N):
    simKern[i]['thetha'] = np.arccos(simKern[i]['px_start'])
    simMantel[i]['thetha'] = np.arccos(simMantel[i]['px_start'])


dfK = deepcopy(simKern[0])
dfM = deepcopy(simMantel[0])

for i in range(49):
    dfK = pd.concat((dfK, simKern[i+1]))
    dfM = pd.concat((dfM, simMantel[i+1]))

nC = 1.60
n1 = 1.49
n2 = 1.42

theta1 = np.arccos(n1/nC)*180/np.pi
theta2 = np.arccos(n2/nC)*180/np.pi
print(f'max angles thetha_1 =  {theta1} thetha_2 = {theta2}')

counts, bins, stf = plt.hist(dfM['thetha']*180/np.pi, bins=50, histtype='step', label='Cladding', linewidth=2, range=(0,50))
plt.hist(dfK['thetha']*180/np.pi, bins=bins, histtype='step', linewidth=2,label='Core');
plt.axvline(x=theta1, label=r'$\theta_1$', linewidth=2, c='tab:red')
plt.axvline(x=theta2, label=r'$\theta_2$', linewidth=2, c='tab:green')
plt.xlabel(r"$\theta$ / Degrees")
plt.ylabel("Counts / a.u.")
plt.legend();
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.tight_layout()
plt.savefig('build/AnglesMantleCore.pdf')
plt.close()


def r_min(y,z,py,pz):
    return np.sqrt(((z*py) - (pz*y))**2)*1/np.sqrt(pz**2 + py**2)



simK = deepcopy(simKern[0])
simM = deepcopy(simMantel[0])

for i in range(N-1):
    simK = pd.concat((simK, simKern[i+1]))
    simM = pd.concat((simM, simMantel[i+1]))

simK['r_min'] = r_min(y=simK['y_start'].to_numpy(),z=simK['z_start'].to_numpy(), py=simK['py_start'].to_numpy(),pz=simK['pz_start'].to_numpy())
simM['r_min'] = r_min(y=simM['y_start'].to_numpy(),z=simM['z_start'].to_numpy(), py=simM['py_start'].to_numpy(),pz=simM['pz_start'].to_numpy())



plt.hist2d(simK['r_min'],simK['thetha']*180/np.pi,bins=(50,50), range=[[0,0.125],[0,50]]);
plt.colorbar()
plt.xlabel(r'$r_{\mathrm{min}}$ / mm')
plt.ylabel(r"$\theta$ / Degrees")
plt.tight_layout();
plt.savefig('build/rminCore.pdf')
plt.close()


plt.hist2d(simM['r_min'],simM['thetha']*180/np.pi,bins=(50,50),range=[[0,0.125],[0,50]]);
plt.colorbar()
plt.xlabel(r'$r_{\mathrm{min}}$ / mm')
plt.ylabel(r"$\theta$ / Degrees")
plt.tight_layout();
plt.savefig('build/rminMantle.pdf')
plt.close()

df = pd.concat((simK, simM))



h = plt.hist2d(df['gpsPosX'],df['thetha']*180/np.pi,bins=(24,89), range=[[100,2400],[0,44.5]]);
plt.colorbar()
plt.xlabel(r'$x_\mathrm{Exc}$ / mm')
plt.ylabel(r"$\theta$ / Degrees")
plt.tight_layout();
plt.savefig('build/Int2d.pdf')
plt.close()


theta = np.zeros(45)
for i in range(45):
    theta[i] = h[2][2*i+1]
theta


I =  np.zeros(24)

thet = 0.5

for j in range(24):
    I[j] = df['gpsPosX'][(df['thetha']*180/np.pi>(thet-0.5)) & (df['thetha']*180/np.pi<(thet+0.5)) & (df['gpsPosX'] == np.linspace(100,2400,24)[j])].count()


def func(x, I_0, a):
    return I_0 * np.exp(-a * x)


p = []
c = []
for i in range(45):
    I =  np.zeros(24)

    I= df[(df['thetha']*180/np.pi>(theta[i]-0.5)) & (df['thetha']*180/np.pi<(theta[i]+0.5))].groupby('gpsPosX').count()['wl'].to_numpy()

    popt, pcov = curve_fit(func, np.linspace(100,2400,24), I, p0=(24000,2*10**-4))
    p.append(popt)
    c.append(pcov)
    if (i%5==0):
        plt.scatter(np.linspace(100,2400,24), I, label=fr'$\theta$ = {theta[i]-0.5}')
        plt.plot(np.linspace(100,2400,24), func(np.linspace(100,2400,24), *popt))

plt.xlabel(r'$x_\mathrm{Exc}$ / mm')
plt.ylabel("Counts / a.u.")
plt.legend();
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.tight_layout()
plt.savefig('build/simfits.pdf')
plt.close()

def func2(x, a1, b):
    return a1/np.cos(x*np.pi/180) + b * np.tan(x*np.pi/180)


a = np.zeros(45)
for i in range(45):
    a[i] = p[i][1]
# popt, pcov = curve_fit(func2, theta, a)
# print(popt,func2(theta, *popt))
plt.scatter(theta, a, label='Fitparameters', marker="x")
plt.plot(theta, func2(theta, a[0]-0.000013, a[0]/2), label='Theory curve', c='tab:orange')
plt.xlabel(r"$\theta$ / Degrees")
plt.ylabel(r'$a$ / $\mathrm{mm}^{-1}$')
plt.legend();
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.tight_layout()
plt.savefig('build/simThetaFit.pdf')
plt.close()


inteDC = []
inteC = []
for i in range(11):
    inteDC.append(pd.read_csv(f'data_local/TheoCanInte/DarkCounts_h={4*i}deg_v=0deg.txt', sep='\t'))

for i in range(10):
    inteC_=[]
    for j in range(20):
        inteC_.append((pd.read_csv(f'data_local/TheoCanInte/Attenuation_h={4*i}deg_v=0deg_x={j*75}mm.txt', sep='\t')-inteDC[i])['C1'].sum())
    inteC.append(inteC_)


color = ['tab:blue',
 'tab:orange',
 'tab:green',
 'tab:red',
 'tab:purple',
 'tab:brown',
 'tab:pink',
 'tab:gray',
 'tab:olive',
 'tab:cyan']
    


p1 = []
c1 = []
p2 = []
c2 = []
for i in range(10):

    popt1, pcov1 = curve_fit(func, xdata=np.linspace(0,1425,20)[0:7], ydata=inteC[i][0:7], p0=(24000,2*10**-4))
    p1.append(popt1)
    c1.append(pcov1)
    popt2, pcov2 = curve_fit(func, xdata=np.linspace(0,1425,20)[8:-1], ydata=inteC[i][8:-1], p0=(24000,2*10**-4))
    p2.append(popt2)
    c2.append(pcov2)
    plt.scatter(np.linspace(0,1425,20), inteC[i], label=fr'$\theta$ = {i*4}',c=color[i])
    plt.plot(np.linspace(0,7*75,40), func(np.linspace(0,7*75,40), *popt1),c=color[i])
    plt.plot(np.linspace(7*75,1425,60), func(np.linspace(7*75,1425,60), *popt2),c=color[i])
plt.xlabel(r'$x_\mathrm{Exc}$ / mm')
plt.ylabel("Counts / a.u.")
plt.legend(loc='upper right');
plt.tight_layout()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig('build/dataFits.pdf')
plt.close()


for i in range(10):
    p[i] = (p1[i] + p2[i])/2
    c[i] = (c1[i] + c2[i])/2


a = np.zeros(10)
for i in range(10):
    a[i] = p[i][1]
popt, pcov = curve_fit(func2, np.linspace(0,36,10), a)

plt.scatter(np.linspace(0,36,10), a, label='Fitparameters', marker="x")
plt.scatter(32, a[8], marker="x", c='tab:red')
# plt.plot(np.linspace(0,40,100), func2(np.linspace(0,40,100), a[0]-0.000013, a[0]/2), label='Theory curve', c='tab:orange')
plt.plot(np.linspace(0,40,100), func2(np.linspace(0,40,100), *popt), label='Fitted curve', c='tab:orange')
plt.xlabel(r"$\theta$ / Degrees")
plt.ylabel(r'$a$ / $\mathrm{mm}^{-1}$')
plt.legend();
plt.tight_layout()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig('build/DataThetaFit.pdf')
plt.close()



angle = np.zeros(9)

a = np.zeros(9)
for i in range(9):
    if i==8:
        a[-1] = p[9][1]
        angle[-1] = np.linspace(0,36,10)[-1]
        break
    a[i] = p[i][1]
    angle[i] = np.linspace(0,36,10)[i]
print(a,angle)
popt, pcov = curve_fit(func2, angle, a)

plt.scatter(angle, a, label='Fitparameters', marker="x")

# plt.plot(np.linspace(0,40,100), func2(np.linspace(0,40,100), a[0]-0.000013, a[0]/2), label='Theory curve', c='tab:orange')
plt.plot(np.linspace(0,40,100), func2(np.linspace(0,40,100), *popt), label='Theory curve', c='tab:orange')
plt.xlabel(r"$\theta$ / Degrees")
plt.ylabel(r'$a$ / $\mathrm{mm}^{-1}$')
plt.legend();
plt.tight_layout()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig('build/DataThetaFitAlt.pdf')
plt.close()


