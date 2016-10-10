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

plt.imshow(data, origin='upper')
#plt.pcolormesh(data)
plt.axis([0,np.shape(data)[1],0,np.shape(data)[0]])
#plt.colorbar()
plt.title("mirrored and zero-padded coloumn density map")
#plt.title("mirrored coloumn density map")

#plt.savefig("brick_mirrored_padded.pdf")
print("About to save as pdf")
plt.show()

