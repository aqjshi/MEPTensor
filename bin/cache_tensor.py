import pandas as pd
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
from wrapper import npy_preprocessor_v4, heat_component, npy_preprocessor_v4_limit

PRECISION = 3

def process_molecule_tensor(args):
    index, inchi, xyz_array, chiral_centers, rotation, resolution_dim = args

    # Map one-hot encoded arrays to atom types
    atom_dict = {1: 'H', 2: 'C', 3: 'N', 4: 'O', 5: 'F'}
    atom_list = []
    atom_coords = []

    for i in range(len(xyz_array)):
        max_idx = np.argmax(xyz_array[i][3:8])
        if max_idx + 1 in atom_dict:
            atom_list.append(atom_dict[max_idx + 1])
            atom_coords.append(xyz_array[i][:3])

    # Center and scale atom coordinates
    atom_coords = np.array(atom_coords)
    atom_coords -= atom_coords.mean(axis=0)

    max_coord = atom_coords.max(axis=0).max()
    if max_coord == 0:
        max_coord = 1  # To avoid division by zero
    atom_coords /= max_coord

    # Initialize the heat tensor
    estimated_heat_tensor = np.zeros((resolution_dim, resolution_dim, resolution_dim))

    # Scale atom coordinates to fit the new resolution
    scaled_atom_coords = (atom_coords + 1) * (resolution_dim / 2)

    # Populate the heat tensor
    for coord in scaled_atom_coords:
        heat_component(estimated_heat_tensor, coord[0], coord[1], coord[2], 1, resolution_dim)

    # Convert tensor values to list of strings joined by space
    tensor_values = " ".join(str(round(estimated_heat_tensor[x, y, z], PRECISION)) 
                             for x in range(resolution_dim) 
                             for y in range(resolution_dim) 
                             for z in range(resolution_dim))

    return {
        'index': index,
        'inchi': inchi,
        'tensor': tensor_values,
    }

def process_molecule_order(resolution_dim):
    order_dim = resolution_dim**3
    tensor_values = " ".join(str(i + 1) for i in range(order_dim))  # values from 1 to order_dim
    return {'tensor': tensor_values}

def construct_tensor_parallel(index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array, resolution_dim):
    pool = Pool(cpu_count())
    args = [(index_array[i], inchi_array[i], xyz_arrays[i], chiral_centers_array[i], rotation_array[i], resolution_dim) 
            for i in range(len(index_array))]
    
    tensor_results = pool.map(process_molecule_tensor, args)
    order_result = process_molecule_order(resolution_dim)
    
    pool.close()
    pool.join()

    # Filter out None results due to errors
    tensor_results = [res for res in tensor_results if res is not None]

    # Create DataFrames
    tensor_df = pd.DataFrame(tensor_results)
    order_df = pd.DataFrame([order_result])  # Create a DataFrame with a single row
    
    return tensor_df, order_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_file", help="The NPY file containing molecule data")
    parser.add_argument("output_file_csv", help="The output CSV file for the modified tensor data")
    parser.add_argument("order_file_csv", help="The output CSV file for the positional encoding data")
    parser.add_argument("resolution", type=int, help="The resolution of the cube (number of cubes per side)")
    args = parser.parse_args()

    index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array = npy_preprocessor_v4(args.npy_file)
    print(len(index_array))
    tensor_df, order_df = construct_tensor_parallel(index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array, int(args.resolution))
    
    tensor_df.to_csv(args.output_file_csv, index=False)
    order_df.to_csv(args.order_file_csv, index=False)

if __name__ == "__main__":
    main()
