#! /usr/bin/env python
import numpy as np
import pdb
from astropy.io import fits
import matplotlib.pyplot as plt
import pyfftw

brick = fits.open("brick.fits")
data = brick[0].data

print("Creating brick.pdf")
plt.imshow(data, origin='upper')
plt.colorbar()
plt.title("coloumn density map")
plt.savefig("brick.pdf")
plt.clf()

data = np.hstack((data, np.fliplr(data)))
data = np.vstack((data, np.flipud(data)))

margin = (np.shape(data)[0] - np.shape(data)[1])/2

print("Creating brick_mirrored.pdf")

plt.imshow(data, origin='upper')
plt.colorbar()
plt.title("mirrored coloumn density map")
plt.savefig("brick_mirrored.pdf")
plt.clf()

data = np.lib.pad(data, ((0,0),(margin, margin)), 'constant', constant_values=(0,0))

print("Creating brick_mirrored_padded.pdf")
plt.imshow(data, origin='upper')
plt.colorbar()
plt.title("mirrored and zero-padded coloumn density map")
plt.savefig("brick_mirrored_padded.pdf")
plt.clf()

f = pyfftw.interfaces.numpy_fft.fft2(data)
fshift = pyfftw.interfaces.numpy_fft.fftshift(f)
fft = abs(fshift)**2

print("Creating brick_fourier_image.pdf")
plt.imshow(np.log10(fft))
plt.title('Fourier image of column density map')
plt.colorbar()
plt.savefig("brick_fourier_image.pdf")
plt.clf()

#Q6

max_dim = 1278
kxy = np.linspace(-max_dim/2, max_dim/2-1, num=max_dim)
k1 = np.zeros((max_dim, max_dim))
k2 = np.zeros((max_dim, max_dim))
for n in range(0, max_dim):
    k1[n,:] = kxy
    k2[:,n] = kxy

# this is our 2d array with absolute value of k vectors
k_image = np.sqrt(k1*k1 + k2*k2)

Pk = np.zeros(max_dim/2 + 1)
k = np.linspace(0, max_dim/2, num=max_dim/2+1)

for ik, k_val in enumerate(k):
    indices_in_k_shell = np.where((k_image >= ik) & (k_image < ik+1))
    Pk[ik] = 2*np.pi * k_val * np.sum(fft[indices_in_k_shell])

print("Creating brick_power_spectrum.pdf")
plt.plot(Pk)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k')
plt.ylabel('P(k)')
plt.savefig("brick_power_spectrum.pdf")
plt.clf()
