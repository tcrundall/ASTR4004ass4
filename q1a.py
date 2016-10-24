import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

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

f = np.fft.fft2(data)
fshift = np.fft.fftshift(f)
#magnitude_spectrum = fshift
magnitude_spectrum = np.log10(np.abs(fshift))
plt.imshow(magnitude_spectrum)
plt.title('Fourier image of column density map')
plt.colorbar()
plt.savefig("fourier-image.pdf")
plt.show()
