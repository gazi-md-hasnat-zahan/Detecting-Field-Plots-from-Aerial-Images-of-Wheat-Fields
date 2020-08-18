import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.util as util
import skimage.color as color
import skimage.filters as filt
import skimage.restoration as rest
import skimage.morphology as morph
import skimage.segmentation as seg
import scipy.ndimage.morphology as morph2
import skimage.external.tifffile as tiff
import skimage.transform as sktr
import skimage.measure as meas
import skimage.exposure as exp
import scipy.misc as m

# Code for image preprocessing from TIFF (big image) to Downscaled image
# Please, comment this after running once
I = tiff.imread('orthomosaic.tif')
# I=util.img_as_float(I)
print('Done')
print(I.shape)
fig1=plt.figure()
fig1.add_subplot(1,2,1)
plt.imshow(I)
I2=sktr.downscale_local_mean(I,(10,10,1))
print(I2.shape)
fig1.add_subplot(1,2,2)
plt.imshow(I2)
plt.show()
m.imsave('downscaled.png',I2)
# End of Code for image preprocessing from TIFF (big image) to Downscaled image

#    DONE    DONE     DONE
I=io.imread('downscaled.png')
I=util.img_as_float(I)
# denoise using gaussian filter
I=filt.gaussian(I,sigma=1,multichannel=True)
m.imsave('downscaled_denoised.png',I)
# convert the colorspace to 2 different grayscale schemes
gndvi = (I[:,:,2]-I[:,:,0])/(I[:,:,2]+I[:,:,0])
custom = 3*I[:,:,0]-I[:,:,1]-I[:,:,2]
fig1=plt.figure()
fig1.add_subplot(1,2,1)
plt.imshow(gndvi)
fig1.add_subplot(1,2,2)
plt.imshow(custom)

c=exp.rescale_intensity(custom)

plt.show()
m.imsave('graylevel_gndvi.png',gndvi)
m.imsave('graylevel_custom.png',custom)
#    DONE    DONE     DONE

# I=io.imread('downscaled_gndvi.png')
# I=util.img_as_float(I)
#  fig1=plt.figure()
# plt.imshow(I)
# plt.show()
