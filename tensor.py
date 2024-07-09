# Qingjian Shi 2024
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse
from wrapper import npy_preprocessor_v3, heat_component, npy_preprocessor_v3_limit

def construct_tensor(index_array, inchi_array, chiral_centers_array, rotation_array, DENSITY):
    tensor_list = []
    inchi_list = []
    chiral_centers_list = []
    rotation0_list = []
    rotation1_list = []
    rotation2_list = []

    for i in range(len(index_array)):
        index = index_array[i]
        inchi = inchi_array[i]
        chiral_centers = len(chiral_centers_array[i])
        rotation = rotation_array[i]

        molecule = Chem.MolFromInchi(inchi)

        # Check if the molecule is valid
        if molecule is None:
            # print(f"Invalid InChI string for index {index}")
            continue

        # Add hydrogens to the molecule
        molecule = Chem.AddHs(molecule)

        # Generate 3D coordinates if they do not exist
        if not molecule.GetNumConformers():
            try:
                AllChem.EmbedMolecule(molecule, randomSeed=42)
                AllChem.UFFOptimizeMolecule(molecule)
            except ValueError as e:
                # print(f"Error generating conformer for index {index}: {e}")
                continue

        # Extract 3D coordinates
        conf = molecule.GetConformer()
        atom_coords = conf.GetPositions()

        # Center and scale atom coordinates
        atom_coords -= atom_coords.mean(axis=0)
        max_coord = atom_coords.max(axis=0).max()
        if max_coord == 0:
            max_coord = 1  # To avoid division by zero
        atom_coords /= max_coord

        # Generate coordinates for the 3D scatter plot and convert to float
        density_dim = DENSITY
        estimated_heat_tensor = np.zeros((density_dim, density_dim, density_dim))

        # Scale atom coordinates to fit the new density
        scaled_atom_coords = (atom_coords + 1) * (density_dim / 2)

        for coord in scaled_atom_coords:
            heat_component(estimated_heat_tensor, coord[0], coord[1], coord[2], 1, density_dim)

        # Convert tensor values to list of strings joined by space
        tensor_values = []
        for x in range(density_dim):
            for y in range(density_dim):
                for z in range(density_dim):
                    tensor_values.append(f"{x} {y} {z} {estimated_heat_tensor[x, y, z]}")
        
        tensor_list.append(" ".join(tensor_values))
        inchi_list.append(inchi)
        chiral_centers_list.append(chiral_centers)
        rotation0_list.append(rotation[0])
        rotation1_list.append(rotation[1])
        rotation2_list.append(rotation[2])

    # Create DataFrame
    df = pd.DataFrame({
        'index': index_array,
        'inchi': inchi_list,
        'tensor_values': tensor_list,
        'chiral_centers': chiral_centers_list,
        'rotation0': rotation0_list,
        'rotation1': rotation1_list,
        'rotation2': rotation2_list
    })

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_file", help="The NPY file containing molecule data")
    args = parser.parse_args()

    index_array, inchi_array, chiral_centers_array, rotation_array = npy_preprocessor_v3(args.npy_file)
    df = construct_tensor(index_array, inchi_array, chiral_centers_array, rotation_array, 9)
    df.to_csv('tensor_dataset_v3.csv', index=False)

if __name__ == "__main__":
    main()
