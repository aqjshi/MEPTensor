# return cache_ohe_order.csv
# cache_ohe_order.csv columns: index, inchi, agent0,agent1,agent2,agent3,agent4,agent5,agent6,agent7,
# agent8,agent9,agent10,agent11,agent12,agent13,agent14,agent15,agent16,agent17,agent18,agent19,agent20,
# agent21,agent22,agent23,agent24,agent25,agent26,agent27,agent28,agent29,agent30,agent31

# agent{index} =  atom_type, chiral_activation, one_dim_position0, one_dim_position1, one_dim_position2, one_dim_position3, 
# one_dim_position4,one_dim_position5,one_dim_position6,one_dim_position7



import numpy as np
import pandas as pd
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
from wrapper import npy_preprocessor_v4, heat_component, npy_preprocessor_v4_limit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PRECISION = 3



# Updated atom dictionary with correct atomic numbers
atom_dict = {1: 'H', 2: 'C', 3: 'N', 4: 'O', 5: 'F'}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process NPY file containing molecule data.")
    parser.add_argument("npy_file", help="The NPY file containing molecule data")
    parser.add_argument("output_file", help="Output filename")
    return parser.parse_args()

def process_ohe_arrays(xyz_arrays, index):
    # ohe_x_array, ohe_y_array, ohe_z_array = [], [], []
    ohe_order = ""

    for xyz_array in xyz_arrays[index]:
        if np.any(xyz_array[3:]):  # Check if any elements in positions 3 to 7 are non-zero
            atom_type_index = np.argmax(xyz_array[3:]) + 1  # Find the index of the first non-zero element
            # if atom_dict[atom_type_index] != 'H':  # Exclude hydrogen atoms
            # ohe_x_array.append(xyz_array[0])
            # ohe_y_array.append(xyz_array[1])
            # ohe_z_array.append(xyz_array[2])
            ohe_order += atom_dict[atom_type_index]  # Convert to atom type using the dictionary

    # return ohe_x_array, ohe_y_array, ohe_z_array, ohe_order
    return ohe_order

def get_rdkit_coordinates(molecule):
    x_array, y_array, z_array = [], [], []
    atom_symbols = []

    if not molecule.GetConformers():  # Check if there are no conformers
        AllChem.EmbedMolecule(molecule)  # Generate conformers
        AllChem.UFFOptimizeMolecule(molecule)  # Optimize the conformers

    conf = molecule.GetConformer()
    for atom in molecule.GetAtoms():
        if atom.GetSymbol() != 'H':  # Exclude hydrogen atoms
            pos = conf.GetAtomPosition(atom.GetIdx())
            x_array.append(pos.x)
            y_array.append(pos.y)
            z_array.append(pos.z)
            atom_symbols.append(atom.GetSymbol())

    return x_array, y_array, z_array, atom_symbols

def main():
    args = parse_arguments()
    index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array = npy_preprocessor_v4(args.npy_file)
    
    index = 0
    inchi = inchi_array[index]
    # ohe_x_array, ohe_y_array, ohe_z_array, ohe_order = process_ohe_arrays(xyz_arrays, index)
     
    longest_ohe_order = 32
    results = []
    
    for i in range(len(index_array)):
        ohe_order = process_ohe_arrays(xyz_arrays, i)
        padded_ohe_order = " ".join(ohe_order.ljust(longest_ohe_order, "0"))  # Pad ohe order
        results.append((index_array[i], inchi_array[i], padded_ohe_order))

    df = pd.DataFrame(results, columns=['Index', 'InChI', 'ohe_order'])
    df.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    main()

