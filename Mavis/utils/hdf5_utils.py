from h5py import File, Group

# Print out the names and shapes of all datasets in an HDF5 file
# Follow groups to the second level
from h5py import File, Group
def print_hdf5_info(file_path):
    with File(file_path, "r") as f:
        print(f.__len__(), "datasets/groups in the file:")
        for name in f:
            if type(f[name]) is Group:
                print("Group name:", name)
                for sub_name in f[name]:
                    print("  Sub-dataset name:", sub_name, " Shape:", f[name][sub_name].shape)
            else:
                print("Dataset name:", name, " Shape:", f[name].shape)

# Use this to merge the all_tokens.hdf5 files from multiple rounds of data collection. This allows the trees to be used as part
# of the same dataset
def merge_hdf5_files(input_files, output_file):
    """
    Merges multiple HDF5 files into a single file. The files must have the same set of groups, if any.
    """
    with File(output_file, "w") as output_f:
        for input_file in input_files:
            with File(input_file, "r") as input_f:
                for name in input_f:
                    if type(name) is Group:
                        if name not in output_f:
                            output_f.create_group(name)
                        for sub_name in input_f[name]:
                            if sub_name not in output_f[name]:
                                output_f[name][sub_name] = input_f[name][sub_name][:]
                    else:
                        if name not in output_f:
                            output_f[name] = input_f[name][:]