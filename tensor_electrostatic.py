import pandas as pd
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
from wrapper import npy_preprocessor_v4

PRECISION = 3

def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def calculate_electrostatic_potential_energy(tensor_positions, atom_coords, nuclear_charges):
    electrostatic_potential_energy = np.zeros(tensor_positions.shape[0])
    ke = 8.9875517873681764 / 1000  # Coulomb's constant in N m²/C²
    
    for i, atom_xyz in enumerate(atom_coords):
        distances = np.linalg.norm(tensor_positions - atom_xyz, axis=1)
        nuclear_charge = nuclear_charges[i]
        electrostatic_potential_energy += np.where(
            distances == 0,
            0.5 * nuclear_charge**2.4,
            ke * nuclear_charge / distances
        )
    return electrostatic_potential_energy

def process_molecule(args):
    index, inchi, xyz_array, chiral_center, rotation, density_dim = args

    atom_dict = {1: 'H', 2: 'C', 3: 'N', 4: 'O', 5: 'F'}
    atom_list = []
    atom_coords = []

    for i in range(len(xyz_array)):
        max_idx = np.argmax(xyz_array[i][3:8])
        if max_idx + 1 in atom_dict:
            atom_list.append(atom_dict[max_idx + 1])
            atom_coords.append(xyz_array[i][:3])

    atom_coords = np.array(atom_coords)
    atom_coords -= atom_coords.mean(axis=0)

    max_coord = atom_coords.max(axis=0).max()
    max_coord = max_coord if max_coord != 0 else 1  # To avoid division by zero
    atom_coords /= max_coord
    atom_coords = (atom_coords + 1) * (density_dim / 2)

    # Create a list of nuclear charges corresponding to the atom_list
    nuclear_charges_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
    nuclear_charges = [nuclear_charges_dict[atom] for atom in atom_list]

    if len(atom_coords) != len(nuclear_charges):
        print(f"Error: Mismatch in number of atom coordinates and nuclear charges for index {index}")
        print(f"Atom coordinates: {atom_coords}")
        print(f"Nuclear charges: {nuclear_charges}")

    x, y, z = np.meshgrid(np.arange(density_dim), np.arange(density_dim), np.arange(density_dim), indexing='ij')
    tensor_positions = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    electrostatic_potential_energy_values = calculate_electrostatic_potential_energy(tensor_positions, atom_coords, nuclear_charges)
    combined_tensor_values = [f"{round(val, PRECISION)}" for val in electrostatic_potential_energy_values]

    return {
        'index': index,
        'inchi': inchi,
        'tensor': " ".join(combined_tensor_values),
        'chiral_length': len(chiral_center),
        'chiral0': chiral_center[0][1] if chiral_center else 0,
        'rotation0': rotation[0],
        'rotation1': rotation[1],
        'rotation2': rotation[2]
    }

def construct_tensor_parallel(index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array, density_dim):
    pool = Pool(cpu_count())
    args = [(index_array[i], inchi_array[i], xyz_arrays[i], chiral_centers_array[i], rotation_array[i], density_dim) for i in range(len(index_array))]

    # for i, arg in enumerate(args):
    #     print(f"Processing molecule {i}: {arg}")

    results = pool.map(process_molecule, args)

    pool.close()
    pool.join()

    results = [res for res in results if res is not None]

    if results:
        keys = results[0].keys()
        for i, res in enumerate(results):
            if res.keys() != keys:
                print(f"Mismatch in keys at index {i}: {res.keys()} vs {keys}")

    df = pd.DataFrame(results)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_file", help="The NPY file containing molecule data")
    parser.add_argument("output_file_csv", help="The output CSV file for the modified tensor data")
    parser.add_argument("density", type=int, help="The density of the cube (number of cubes per side)")
    args = parser.parse_args()

    index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array = npy_preprocessor_v4(args.npy_file)
    print(len(index_array))
    df = construct_tensor_parallel(index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array, int(args.density))
    df.to_csv(args.output_file_csv, index=False)

    print(f"Electrostatic potential energy tensor data saved to {args.output_file_csv}")

if __name__ == "__main__":
    main()
