import pandas as pd
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from wrapper2 import npy_preprocessor_v4, heat_component, npy_preprocessor_v4_limit

PRECISION = 3

atom_dict = {1: 'H', 2: 'C', 3: 'N', 4: 'O', 5: 'F'}

def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def calculate_electrostatic_potential_energy(tensor_positions, atom_coords, nuclear_charges):
    electrostatic_potential_energy = np.zeros(tensor_positions.shape[0])
    ke = 8.9875517873681764  /1000 # Coulomb's constant in N m²/C²

    for i, atom_xyz in enumerate(atom_coords):
        distances = np.array([distance(tp, atom_xyz) for tp in tensor_positions])
        nuclear_charge = nuclear_charges[i]
        electrostatic_potential_energy += np.where(
            distances < 1e-8,
            0.5 * nuclear_charge**2.4,
            ke * nuclear_charge / distances
        )
    return electrostatic_potential_energy

def process_molecule(args):
    index, inchi, xyz_array, chiral_centers, rotation, resolution_dim = args

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
    estimated_heat_tensor = np.zeros((resolution_dim, resolution_dim, resolution_dim))

    # Scale atom coordinates to fit the new resolution
    scaled_atom_coords = (atom_coords + 1) * (resolution_dim / 2)

    # Populate the heat tensor
    for coord in scaled_atom_coords:
        heat_component(estimated_heat_tensor, coord[0], coord[1], coord[2], 1, resolution_dim)

    # Calculate electrostatic potential energy
    x, y, z = np.meshgrid(np.arange(resolution_dim), np.arange(resolution_dim), np.arange(resolution_dim), indexing='ij')
    tensor_positions = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    # Create a list of nuclear charges corresponding to the atom_list
    nuclear_charges_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
    nuclear_charges = [nuclear_charges_dict[atom] for atom in atom_list]

    electrostatic_potential_energy_values = calculate_electrostatic_potential_energy(tensor_positions, scaled_atom_coords, nuclear_charges)
    combined_tensor_values = [f"{round(val, PRECISION)}" for val in electrostatic_potential_energy_values]

    return {
        'index': index,
        'inchi': inchi,
        'tensor': " ".join(combined_tensor_values),
        'chiral_length': len(chiral_centers),
        'chiral0': chiral_centers[0][1] if chiral_centers else 0,
        'rotation0': rotation[0],
        'rotation1': rotation[1],
        'rotation2': rotation[2]
    }

def construct_tensor_parallel(index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array, resolution_dim):
    # Find min and max values for each axis
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
    args = [(index_array[i], inchi_array[i], xyz_arrays[i], chiral_centers_array[i], rotation_array[i], resolution_dim)
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
    parser.add_argument("output_file_csv", help="The output CSV file for the modified tensor data")
    parser.add_argument("resolution", type=int, help="The resolution of the cube (number of cubes per side)")
    args = parser.parse_args()

    # expect df['index'].values[:limit], df['inchi'].values[:limit], df['xyz'].values[:limit], df['chiral_centers'].values[:limit], df['rotation'].values[:limit]
    index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array = npy_preprocessor_v4(args.npy_file)
    print(len(index_array))
    df = construct_tensor_parallel(index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array, int(args.resolution))
    df.to_csv(args.output_file_csv, index=False)

    print(f"Electrostatic potential energy tensor data saved to {args.output_file_csv}")

if __name__ == "__main__":
    main()
