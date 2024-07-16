import pandas as pd
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
from wrapper import npy_preprocessor_v4, heat_component, npy_preprocessor_v4_limit

PRECISION = 3

def process_molecule(args):
    index, inchi, xyz_array, chiral_centers, rotation, density_dim = args

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
    estimated_heat_tensor = np.zeros((density_dim, density_dim, density_dim))

    # Scale atom coordinates to fit the new density
    scaled_atom_coords = (atom_coords + 1) * (density_dim / 2)

    # Populate the heat tensor
    for coord in scaled_atom_coords:
        heat_component(estimated_heat_tensor, coord[0], coord[1], coord[2], 1, density_dim)

    # Convert tensor values to list of strings joined by space
    tensor_values = " ".join(str(round(estimated_heat_tensor[x, y, z], PRECISION)) 
                             for x in range(density_dim) 
                             for y in range(density_dim) 
                             for z in range(density_dim))

    # Determine the chirality of the first chiral center, if it exists
    chiral0 = chiral_centers[0][1] if chiral_centers else '0'

    return {
        'index': index,
        'inchi': inchi,
        'tensor': tensor_values,
        'chiral_length': len(chiral_centers),
        'chiral0': chiral0,
        'rotation0': rotation[0],
        'rotation1': rotation[1],
        'rotation2': rotation[2]
    }

def construct_tensor_parallel(index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array, density_dim):
    pool = Pool(cpu_count())
    args = [(index_array[i], inchi_array[i], xyz_arrays[i], chiral_centers_array[i], rotation_array[i], density_dim) 
            for i in range(len(index_array))]
    
    results = pool.map(process_molecule, args)
    pool.close()
    pool.join()

    # Filter out None results due to errors
    results = [res for res in results if res is not None]

    # Create DataFrame
    df = pd.DataFrame(results)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_file", help="The NPY file containing molecule data")
    args = parser.parse_args()

    index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array = npy_preprocessor_v4(args.npy_file)
    df = construct_tensor_parallel(index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array, 9)
    df.to_csv('tensor_dataset_v3.csv', index=False)

if __name__ == "__main__":
    main()
