import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse
from multiprocessing import Pool, cpu_count
from wrapper import npy_preprocessor_v3, heat_component, npy_preprocessor_v3_limit

def process_molecule(args):
    index, inchi, chiral_centers, rotation, density_dim = args

    molecule = Chem.MolFromInchi(inchi)

    # Check if the molecule is valid
    if molecule is None:
        # print(f"Invalid InChI string for index {index}")
        return None

    # Add hydrogens to the molecule
    molecule = Chem.AddHs(molecule)

    # Generate 3D coordinates if they do not exist
    if not molecule.GetNumConformers():
        try:
            AllChem.EmbedMolecule(molecule, randomSeed=42)
            AllChem.UFFOptimizeMolecule(molecule)
        except ValueError as e:
            # print(f"Error generating conformer for index {index}: {e}")
            return None

    # Extract 3D coordinates
    conf = molecule.GetConformer()
    atom_coords = conf.GetPositions()

    # Center and scale atom coordinates
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
    tensor_values = []
    for x in range(density_dim):
        for y in range(density_dim):
            for z in range(density_dim):
                tensor_values.append(str(estimated_heat_tensor[x, y, z]))  # Convert to string

    return {
        'index': index,
        'inchi': inchi,
        'tensor_values': " ".join(tensor_values),
        'chiral_centers': len(chiral_centers),
        'rotation0': rotation[0],
        'rotation1': rotation[1],
        'rotation2': rotation[2]
    }

def construct_tensor_parallel(index_array, inchi_array, chiral_centers_array, rotation_array, DENSITY):
    density_dim = DENSITY
    pool = Pool(cpu_count())

    args = [(index_array[i], inchi_array[i], chiral_centers_array[i], rotation_array[i], density_dim) for i in range(len(index_array))]

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

    index_array, inchi_array, chiral_centers_array, rotation_array = npy_preprocessor_v3(args.npy_file)
    df = construct_tensor_parallel(index_array, inchi_array, chiral_centers_array, rotation_array, 9)
    df.to_csv('tensor_dataset_v3.csv', index=False)

if __name__ == "__main__":
    main()
