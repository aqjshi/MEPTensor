import pandas as pd
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from wrapper import npy_preprocessor_v4, heat_component, npy_preprocessor_v4_limit

PRECISION = 3

atom_dict = {1: 'H', 2: 'C', 3: 'N', 4: 'O', 5: 'F'}
  
def process_molecule(args):
    index, inchi, xyz_array, chiral_centers, rotation, density_dim, min_x, min_y, min_z, molecule_range = args

    # Map one-hot encoded arrays to atom types
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
    #find min and max values for each axis
    min_x = np.inf
    min_y = np.inf
    min_z = np.inf
    max_x = -np.inf
    max_y = -np.inf
    max_z = -np.inf
    for xyz_array in xyz_arrays:
        for atom in xyz_array:
            x, y, z = atom[:3]
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            min_z = min(min_z, z)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            max_z = max(max_z, z)
    molecule_range = max(max_x - min_x, max_y - min_y, max_z - min_z)
    
    pool = Pool(cpu_count())
    args = [(index_array[i], inchi_array[i], xyz_arrays[i], chiral_centers_array[i], rotation_array[i], density_dim, min_x, min_y, min_z, molecule_range) 
    for i in range(len(index_array))]
    
    results = pool.map(process_molecule, args)
    pool.close()
    pool.join()

    # Filter out None results due to errors
    results = [res for res in results if res is not None]

    # Create DataFrame
    df = pd.DataFrame(results)
    return df

def plot_molecule(atom_coords, atom_types):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = {'H': 'lightgray', 'C': 'black', 'N': 'blue', 'O': 'red', 'F': 'green'}
    
    for atom, coord in zip(atom_types, atom_coords):
        ax.scatter(coord[0], coord[1], coord[2], color=colors[atom], label=atom)
    
    ax.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_file", help="The NPY file containing molecule data")
    args = parser.parse_args()

    index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array = npy_preprocessor_v4_limit(args.npy_file, limit = 1000)
    df = construct_tensor_parallel(index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array, 9)
    df.to_csv('tensor_dataset_v3.csv', index=False)

if __name__ == "__main__":
    main()
