import h5py

# take file with small time steps and store every n-th entry to make it git-compatible
files = ["infos_step4e-3_dt1e-6_centergrid.hdf5"]
store_every_nth = 10

for file in files:
    name = file.split('.')
    new_name = name[0] + "_slimmed.hdf5"

    with h5py.File(new_name, 'w') as new_file:
        with h5py.File(file, 'r') as old_file:
            for key in old_file.keys():
                # new_file.create_dataset(key, dtype='f8')
                new_file[key] = old_file[key][::store_every_nth]
