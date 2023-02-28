import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
import pandas as pd

def rdown(x, a, b, c, d):
    val=b*(np.exp(-c*(x-a)))+d
    return np.where(val>b,b,val)
    return np.piecewise(x,[x<a,x>=a],[lambda x:b,lambda x:d + b*(np.exp(-c*(x-a)))])

def chi2red(y, yfit, err, ndeg):
    return np.sum(np.divide(np.square(np.subtract(y,yfit)),np.square(err))) / ndeg

exp = [2,4,7,9,12,14,16,19,21,23,25,27,32,34,36,38]


c = [] 
dc = []
for i in exp:
    with open(f"20230223-0002/20230223-0002_{i:02d}.csv") as f:
        contents = f.readlines()
        times = []
        volts = []
        for line in contents[3:]:
            a = line.strip().split(",", 3)
            times.append(float(".".join(a[:2])))
            volts.append(float(".".join(a[2:])))
        times = np.array(times)
        volts = np.array(volts)
        times_un = np.unique(times)

        err = np.multiply(volts, 0.1)
        err = np.add(err, 3)

        a, b = op.curve_fit(rdown,times,volts,p0 = [0,170,1,25], sigma=err)
        plt.scatter(times, volts, s=5)
        #plt.errorbar(times, volts, err, alpha= 0.01, color="r", capsize=3,  lw = 0, markeredgewidth=10)
        #plt.fill_between(times, volts-err, volts+err)
        #plt.plot(times_un,rdown(times_un,*a), color="k", label="fit")
        #plt.xlabel(r"time in $\mu$s")
        #plt.ylabel("volts in mV")
        #plt.legend()
        #plt.savefig(f"ringdown_{i}.png")
        plt.clf()
        c.append(a[2])
        dc.append(b[2][2]**2)
        #print(f"{a[0]:.9f} $\pm$ { (np.diag(b)**2)[0]:.1e} &{a[1]:.5f} $\pm$ { (np.diag(b)**2)[1]:.1e} &{a[2]:.11f} $\pm$ { (np.diag(b)**2)[2]:.1e} & {a[3]:.7f} $\pm$ { (np.diag(b)**2)[3]:.1e}& {chi2red(volts, rdown(times, *a),err, len(volts)-4):.02f}\\\\")

print(f"meanc {np.mean(c)} pm {np.mean(dc)}")
print(f"r = {np.exp(-0.99*np.mean(c)/(2*3e8))} pm {np.exp(-0.99*np.mean(c)/(2*3e8))*np.sqrt(0.99*np.mean(dc)**2/(2*3e8) + 0.005**2*np.mean(c)/(2*3e8))}")
r = np.exp(-0.99*np.mean(c)/(2*3e8))
print(f"F = {np.pi* np.sqrt(np.exp(-0.99*np.mean(c)/(2*3e8))) / (1- np.exp(-0.99*np.mean(c)/(2*3e8)))} pm {np.exp(-0.99*np.mean(c)/(2*3e8))*np.sqrt(0.99*np.mean(dc)**2/(2*3e8) + 0.005**2*np.mean(c)/(2*3e8)) * (0.5*np.pi/(np.sqrt(r)*(1-r)) )}")