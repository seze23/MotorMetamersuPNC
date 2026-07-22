import numpy as np

# Open the file using a context manager
with np.load('/home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/centerout/center_out_0_right_spindles.npz') as data:
    # 1. List all available array names (keys) inside the archive
    print("Arrays in file:", data.files)
    
    # 2. Access a specific array using its key name
    # (By default, keys are usually 'arr_0', 'arr_1', etc., unless named during saving)
    my_array = data['firing_rates']
    print("Array shape:", my_array.shape)
    print("Number of dimensions:", my_array.ndim)
    print(my_array)
