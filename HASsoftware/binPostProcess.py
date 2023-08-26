# Read MCMC output from a file, then compute and plot the auto-covariance function

import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import acor
import ChainOutput
import time

# performance of post-processing improved by reading the MCMC output from a binary file

chain_test      = True
thetas_test     = True

start = time.time()
# read chain from binary file, cast it to vector called X
dtype = np.dtype('float64')
try:
   with open("chain.bin", "rb") as f:
      X = np.fromfile(f,dtype)
except IOError:
   chain_test = False
   
   
# read thetas from binary file, cast it to vector called theta
dtype = np.dtype('double')
try:
   with open("thetas.bin", "rb") as f:
       theta = np.fromfile(f,dtype)
       theta = theta[ theta != 0 ]
except IOError:
   thetas_test = False
   
   
   
   
Ns = ChainOutput.Ns
d  = ChainOutput.d
X = np.reshape(X, (Ns,d))
end = time.time()

print("---------------------------------------------------")
OutputLine = " chain has " + str(Ns) + " rows and " + str(d) + " columns "
print(OutputLine)
print(" Time to read chain from binary =  %.4f seconds" %(end-start))
eps = ChainOutput.eps
beta = 1.0 / (2.0*eps*eps)
ac  = acor.acov(X[:,0])
tau = acor.act(ac, Nmin = 20, w = 7)
print(" tau = %.2f" %tau)
print("---------------------------------------------------")

np1 = 5*int(tau)

#-------------------------------Plot of Autocovaroiance-------------------------
if (chain_test==True):
   fig, ax = plt.subplots()
   ax.plot(range(np1), ac[0:np1], 'bo', label = 'auto-covariance')
   ax.set_ylabel('covariance')
   ax.set_xlabel('lag')
   title = ChainOutput.ModelName
   ax.set_title(title)
   ax.grid()

   ymin, ymax = ax.get_ylim()
   tauLabel = 'tau = {0:8.2f}'.format(tau)
   ax.vlines(tau,  ymin, ymax, label = tauLabel, colors='r')
   ax.legend()

   runInfo = r"$\beta=\frac{1}{2\varepsilon^2}=%.1f$" % (beta) + ", "
   runInfo = runInfo + r"$\varepsilon=%.3f$" % (ChainOutput.eps) + "\n"
   runInfo = runInfo + r"$N_{soft}=%d$" % (ChainOutput.Nsoft) + ", "
   runInfo = runInfo + r"$N_{rattle}=%d$" % (ChainOutput.Nrattle) + "\n"
   runInfo = runInfo + r"$s_s=%.2f$" % (ChainOutput.ks) + r"$\varepsilon$" + ", "
   runInfo = runInfo + r"$T=%d$" % (ChainOutput.T) + "\n"
   runInfo = runInfo + r"$N_s=%d$" % (Ns) + ", "
   runInfo = runInfo + r"$\frac{N_s}{T}=%.2f$" % (float(Ns)/float(ChainOutput.T)) + "\n"
   runInfo = runInfo + r"$A_s=%.3f$" % (ChainOutput.As) + ", "
   runInfo = runInfo + r"$A_r=%.3f$" % (ChainOutput.Ar)

   # these are matplotlib.patch.Patch properties
   props = dict(boxstyle='round', facecolor='white', alpha=0.8)

   # place a text box in upper left in axes coords
   ax.text(0.55, 0.8, runInfo, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)


   plt.savefig('AutoCovariance.pdf')
   #plt.show()


#-------------------------------Theta histogram-----------------------------
if (thetas_test==True):
   ntbins = 100
   truepdf   = 1. / (math.pi)

   fig, ax = plt.subplots(figsize=(10,12))
   ax.hist(theta, ntbins, density=True, alpha=0.5, histtype='bar', ec='black', label=r"$\theta$ sample distr.")
   ax.set_ylabel('Density', fontsize=25)
   ax.set_xlabel(r'$\theta$', fontsize=25)
   ax.set_xlim(left=theta.min(), right=theta.max())
   plt.axhline(y = truepdf, color = 'r', linestyle = '-', label=r"$\theta$ theoretical pdf")
   ax.legend(bbox_to_anchor=(0.755, 1.14), fontsize=22)
   ax.tick_params(axis='both', which='major', labelsize=15)
   ax.tick_params(axis='both', which='minor', labelsize=15)

   plt.savefig('ThetaCount.pdf')



with open('output', 'w') as OutputFile:
   OutputFile.write('estimated tau is {0:8.2f}'.format(tau))



# Check that values agree with old PostProcess.py
#for i in range(0, 20):
   #for j in range(0,3):
      #print('X[    %.d' %i +',    %.d' %j + '] = %.8f' %X[i,j] )

