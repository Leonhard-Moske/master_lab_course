import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import allantools as at
import scipy.optimize as opt

df = pd.read_csv("Raw DataJul1hr.csv")

# function to plot raw data and save it to a file 
def plotRawData(df, name):  
    plt.plot(df["Time (s)"][10:1000], df["Gyroscope x (rad/s)"][10:1000], label = "x")
    #plt.plot(df["Time (s)"][10:1000], df["Gyroscope y (rad/s)"][10:1000], label = "y")
    #plt.plot(df["Time (s)"][10:1000], df["Gyroscope z (rad/s)"][10:1000], label = "z")
    plt.xlabel("Time (s)")
    plt.ylabel("Rotation (rad/s)")
    #plt.yscale('log')
    plt.legend()
    plt.savefig(name)
    plt.clf()

plotRawData(df, "RawData.png")

deltat = df['Time (s)'][1] - df['Time (s)'][0]
freq = np.fft.fftfreq(len(df['Time (s)']), deltat)

x = np.abs(np.fft.fft(df["Gyroscope x (rad/s)"]))
y = np.abs(np.fft.fft(df["Gyroscope y (rad/s)"]))
z = np.abs(np.fft.fft(df["Gyroscope z (rad/s)"]))



plt.plot(freq, x)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Frequency (Hz)")
plt.ylabel("x rotation (rad/s)")
plt.savefig("xFreq.png")
plt.clf()

plt.plot(freq, y)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Frequency (Hz)")
plt.ylabel("y rotation (rad/s)")
plt.savefig("yFreq.png")
plt.ylabel
plt.clf()


plt.plot(freq, z)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Frequency (Hz)")
plt.ylabel("z rotation (rad/s)")
plt.savefig("zFreq.png")
plt.clf()



# allan deviation

# def allandev(data, tau, deltat):
#     data = np.array(data)
#     time = deltat * len(data)
#     M = int(time/tau)
#     N = int(len(data)/M)
#     data = data[0:N*M]
#     data = np.reshape(data, (N, M))
#     data = np.mean(data, axis=1)
#     data = data[1:] - data[:-1]
#     data = data**2
#     return np.mean(data) / 2


t = np.linspace(0, 60*60/2, 60*60//2)
r = 1/deltat

allanx = at.adev(np.array(df["Gyroscope x (rad/s)"]), rate=r,data_type="freq", taus = t)
allany = at.adev(np.array(df["Gyroscope y (rad/s)"]), rate=r,data_type="freq", taus = t)
allanz = at.adev(np.array(df["Gyroscope z (rad/s)"]), rate=r,data_type="freq", taus = t)

# plt.plot(allanx[0], allanx[1])
# plt.xscale('log')
# plt.yscale('log')
# plt.show()

# plt.plot(allany[0], allany[1])
# plt.xscale('log')
# plt.yscale('log')
# plt.show()

# plt.plot(allanz[0], allanz[1])
# plt.xscale('log')
# plt.yscale('log')
# plt.show()

# fit to allan deviation

def func(x, a):
    return a/np.sqrt(x)

maxx = np.argmin(allanx[1])
maxy = np.argmin(allany[1])
maxz = np.argmin(allanz[1])


poptx, pcovx = opt.curve_fit(func, allanx[0][0:maxx], allanx[1][0:maxx], full_output=True)
popty, pcovy = opt.curve_fit(func, allany[0][0:maxy], allany[1][0:maxy])
poptz, pcovz = opt.curve_fit(func, allanz[0][0:maxz], allanz[1][0:maxz])


plt.plot(allanx[0], allanx[1])
plt.plot(allanx[0][0:maxx],func(allanx[0][0:maxx], *poptx), label = f"A = {poptx[0]:.04} +/- {np.sqrt(pcovx[0][0]):.04}")
plt.xlabel(r"$\tau$ (s)")
plt.ylabel(r"$\sigma_x(\tau)$")
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.savefig("xAllan.png")
plt.clf()


plt.plot(allany[0], allany[1])
plt.plot(allany[0][0:maxy],func(allany[0][0:maxy], *popty) , label = f"A = {popty[0]:.04} +/- {np.sqrt(pcovy[0][0]):.04}")
plt.xlabel(r"$\tau$ (s)")
plt.ylabel(r"$\sigma_y(\tau)$")
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.savefig("yAllan.png")
plt.clf()


plt.plot(allanz[0], allanz[1])
plt.plot(allanz[0][0:maxz],func(allanz[0][0:maxz], *poptz),label = f"A = {poptz[0]:.04} +/- {np.sqrt(pcovz[0][0]):.04}")
plt.xlabel(r"$\tau$ (s)")
plt.ylabel(r"$\sigma_z(\tau)$")
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.savefig("zAllan.png")
plt.clf()
