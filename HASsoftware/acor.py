# Calculate the auto-covariance function of a sequence
# Use FFT routines from Scipy

#  Written by Jonathan Goodman, May, 2021 (for the n-th time!)

import numpy as np
import scipy.fft as fft

def acov(X):
   """Compute and return the auto-covariance function of a sequence
      input is X: a one index numpy array of reals
      output (return value): a one index numpy array of reals, with the
      same length as the input."""
      
   T       = X.size
   Xbar    = np.sum(X)/T
   Xd      = np.zeros(2*T)  # The padded series -- double length
   Xd[0:T] = X[:]-Xbar      # Remove the mean from the X values, leave the padding as zero
   
   Xhat   = fft.fft(Xd)
   mod2   = lambda z: np.real(z)**2 + np.imag(z)**2  # mod2(z) = |z|^2
   Xhsq   = mod2(Xhat)
   acc    = np.real( fft.ifft(Xhsq) ) # The FFT of the auto-covariance is the square modulus
   acov   = acc[0:T]                  #  The first half if real.  The second half is junk from padding.
   for t in range(T-1):
      acov[t] = acov[t]/(T-t-1)       # normalize to estimate the covariance
   return acov
   
def act(acov, w=5., Nmin = 10):
   """Estimate the auto-correlation time, tau,  from a previously computed
      auto-covariance function.
      acov: previously computed auto-covariance function (use the function acov)
      w:    size of self consistent window -- sum to t = w*tau
      Nmin: minimum allowed effective sample size.  Complain if T < Nmin*tau"""
   
   T   = acov.size
   sv  = acov[0]      # the static variance is the lag zero covariance
   tau = 1.
   
   for t in range(1,(T-1)):     #  the last value of acov is junk
      if ( t > w*tau):
         break
      tau += 2*acov[t]/sv
   if ( t > w*tau):
      if (T > Nmin*tau):
         return tau
      else:
         OutputLine = "acor.act complaining about short time series with "
         OutputLine = OutputLine + "estimated tau = " + str(tau)
         OutputLine = OutputLine + ", and T = " + str(T)
         print( OutputLine )
         return
   
   
   
