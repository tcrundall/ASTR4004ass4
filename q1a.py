import numpy as np
import pdb
from astropy.io import fits
import matplotlib.pyplot as plt
import pyfftw

brick = fits.open("brick.fits")

print(brick.info())

data = brick[0].data

print(np.shape(data))
# 638  x 429
# vert x horz

data = np.hstack((data, np.fliplr(data)))
data = np.vstack((data, np.flipud(data)))
print(np.shape(data))

margin = (np.shape(data)[0] - np.shape(data)[1])/2
print(margin)
data = np.lib.pad(data, ((0,0),(margin, margin)), 'constant', constant_values=(0,0))

#plt.imshow(data, origin='upper')
#plt.pcolormesh(data)
#plt.axis([0,np.shape(data)[1],0,np.shape(data)[0]])
#plt.colorbar()
#plt.title("mirrored and zero-padded coloumn density map")
#plt.title("mirrored coloumn density map")

#plt.savefig("b2rick_mirrored_padded.pdf")
print("About to save as pdf")
#plt.show()

f = pyfftw.interfaces.numpy_fft.fft2(data)
fshift = pyfftw.interfaces.numpy_fft.fftshift(f)
#magnitude_spectrum = fshift
#magnitude_spectrum = np.log10(np.abs(f))
fft = abs(fshift)**2

plt.imshow(np.log10(fft))
plt.title('Fourier image of column density map')
plt.colorbar()
#plt.savefig("fourier-image.pdf")
plt.show()

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

plt.imshow(k_image)
plt.show()

Pk = np.zeros(max_dim/2 + 1)
k = np.linspace(0, max_dim/2, num=max_dim/2+1)

for ik, k_val in enumerate(k):
    indices_in_k_shell = np.where((k_image >= ik) & (k_image < ik+1))
    Pk[ik] = 2*np.pi * k_val * np.sum(fft[indices_in_k_shell])

plt.plot(Pk)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k')
plt.ylabel('P(k)')
plt.show()
plt.savefig("brick_power_spectrum.pdf")
