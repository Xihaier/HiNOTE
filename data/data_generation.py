import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt


#####################
# weather data
#####################
_file = h5py.File('data/era5/valid_2/2007.h5', 'r')
data_2007 = _file['fields']
_file = h5py.File('data/era5/train/2008.h5', 'r')
data_2008 = _file['fields']
_file = h5py.File('data/era5/test_1/2009.h5', 'r')
data_2009 = _file['fields']
_file = h5py.File('data/era5/train/2010.h5', 'r')
data_2010 = _file['fields']
_file = h5py.File('data/era5/train/2011.h5', 'r')
data_2011 = _file['fields']
_file = h5py.File('data/era5/valid_1/2012.h5', 'r')
data_2012 = _file['fields']

idx = 2
dat = np.concatenate((data_2007[:,idx,:,:], data_2008[:,idx,:,:], data_2009[:,idx,:,:], data_2010[:,idx,:,:], data_2011[:,idx,:,:], data_2012[:,idx,:,:]), axis=0)
print(dat.shape)

np.save('water_vapor.npy', dat)

dat = np.load("kinetic_energy.npy")
dat = dat[:,:720,:]
print(dat.shape)
np.save("kinetic_energy.npy", dat)

dat = np.load("temperature.npy")
dat = dat[:,:720,:]
print(dat.shape)
np.save("temperature.npy", dat)

dat = np.load("water_vapor.npy")
dat = dat[:,:720,:]
print(dat.shape)
np.save("water_vapor.npy", dat)


#####################
# fluid and cosmology data
#####################
data_dir = "data/era5/train"
files_paths = glob.glob(data_dir + "/*.h5")
files_paths.sort()

with h5py.File(files_paths[0], 'r') as _f:
    print("Getting file stats from {}".format(files_paths[0]))
    n_samples_per_file = _f['fields'].shape[0]
    n_in_channels = _f['fields'].shape[1]
    img_shape_x = _f['fields'].shape[2]
    img_shape_y = _f['fields'].shape[3]
    _file = h5py.File(files_paths[0], 'r')
    files = _file['fields']  
    print(files.shape)

    dat1 = files[0,0,:,:]
    dat2 = files[0,1,:,:]
    dat3 = files[0,2,:,:]

dat = files[:,1,:,:]
print(dat.shape)

np.save('baryon_density.npy', dat)


#####################
# visualization
#####################
cmap = plt.get_cmap("RdBu_r")
plt.close("all")
fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
ax0.set_title("X Velocity")
cset1 = ax0.imshow(dat1, cmap=cmap)
ax0.set_xticks([], [])
ax0.set_yticks([], [])
fig.colorbar(cset1, ax=ax0)
ax1.set_title("Y Velocity")
cset2 = ax1.imshow(dat2, cmap=cmap)
ax1.set_xticks([], [])
ax1.set_yticks([], [])
fig.colorbar(cset2, ax=ax1)
ax2.set_title("Vorticity")
cset3 = ax2.imshow(dat3, cmap=cmap)
ax2.set_xticks([], [])
ax2.set_yticks([], [])
fig.colorbar(cset3, ax=ax2)
plt.savefig("sample_ns.png", bbox_inches="tight")

cmap = plt.get_cmap("RdBu_r")
plt.close("all")
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax0.set_title("Temperature")
cset1 = ax0.imshow(dat1, cmap=cmap)
ax0.set_xticks([], [])
ax0.set_yticks([], [])
fig.colorbar(cset1, ax=ax0)
ax1.set_title("Baryon density")
cset2 = ax1.imshow(dat2, cmap=cmap)
ax1.set_xticks([], [])
ax1.set_yticks([], [])
fig.colorbar(cset2, ax=ax1)
plt.savefig("sample_cos.png", bbox_inches="tight")

cmap = plt.get_cmap("RdBu_r")
plt.close("all")
fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
ax0.set_title("Kinetic Energy")
cset1 = ax0.imshow(dat1, cmap=cmap)
ax0.set_xticks([], [])
ax0.set_yticks([], [])
fig.colorbar(cset1, ax=ax0)
ax1.set_title("Temperature")
cset2 = ax1.imshow(dat2, cmap=cmap)
ax1.set_xticks([], [])
ax1.set_yticks([], [])
fig.colorbar(cset2, ax=ax1)
ax2.set_title("Water Vapor")
cset3 = ax2.imshow(dat3, cmap=cmap)
ax2.set_xticks([], [])
ax2.set_yticks([], [])
fig.colorbar(cset3, ax=ax2)
plt.savefig("sample_weather.png", bbox_inches="tight")