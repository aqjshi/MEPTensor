# python bin/cache_ohe_order.py qm9_filtered.npy cache/cache_ohe_order.csv 9

# output cache/cache_ohe_order.csv

# cache/cache_ohe_order.csv columns: index, inchi, agent0,agent1,agent2,agent3,agent4,agent5,agent6,agent7,
# agent8,agent9,agent10,agent11,agent12,agent13,agent14,agent15,agent16,agent17,agent18,agent19,agent20,
# agent21,agent22,agent23,agent24,agent25,agent26,agent27,agent28,agent29,agent30,agent31

# agent{index} = atom_type, chiral_activation, true_x, true_y, true_z, one_dim_position0, one_dim_position1, one_dim_position2, one_dim_position3,
# one_dim_position4,one_dim_position5,one_dim_position6,one_dim_position7

import numpy as np
import pandas as pd
import argparse
from multiprocessing import Pool, cpu_count
from rdkit import Chem
from rdkit.Chem import AllChem
from wrapper import npy_preprocessor_v4_limit, npy_preprocessor_v4

PRECISION = 3
atom_dict = {1: 'H', 2: 'C', 3: 'N', 4: 'O', 5: 'F'}

def agent_heat_component(tensor, tensor_x, tensor_y, tensor_z, tensor_weight, resolution_dim):
    # Determine the base indices
    base_x, base_y, base_z = int(tensor_x), int(tensor_y), int(tensor_z)
    # Get the decimal part for bleeding
    bleed_x = tensor_x % 1
    bleed_y = tensor_y % 1
    bleed_z = tensor_z % 1

    complement_bleed_x = 1 - bleed_x
    complement_bleed_y = 1 - bleed_y
    complement_bleed_z = 1 - bleed_z

    # Calculate the weights for each of the neighboring cells
    weights = np.zeros((2, 2, 2))
    weights[0, 0, 0] = complement_bleed_x * complement_bleed_y * complement_bleed_z * tensor_weight
    weights[0, 0, 1] = complement_bleed_x * complement_bleed_y * bleed_z * tensor_weight
    weights[0, 1, 0] = complement_bleed_x * bleed_y * complement_bleed_z * tensor_weight
    weights[0, 1, 1] = complement_bleed_x * bleed_y * bleed_z * tensor_weight
    weights[1, 0, 0] = bleed_x * complement_bleed_y * complement_bleed_z * tensor_weight
    weights[1, 0, 1] = bleed_x * complement_bleed_y * bleed_z * tensor_weight
    weights[1, 1, 0] = bleed_x * bleed_y * complement_bleed_z * tensor_weight
    weights[1, 1, 1] = bleed_x * bleed_y * bleed_z * tensor_weight

    applied_positions = []

    # Add weights to the tensor
    for dx in range(2):
        for dy in range(2):
            for dz in range(2):
                if 0 <= base_x + dx < tensor.shape[0] and 0 <= base_y + dy < tensor.shape[1] and 0 <= base_z + dz < tensor.shape[2]:
                    tensor[base_x + dx, base_y + dy, base_z + dz] += weights[dx, dy, dz]
                    applied_positions.append((base_x + dx) * resolution_dim**2 + (base_y + dy) * resolution_dim + (base_z + dz))

    # Pad applied_positions to ensure it has exactly 8 elements
    while len(applied_positions) < 8:
        applied_positions.append(-1)

    return tensor, applied_positions[:8]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process NPY file containing molecule data.")
    parser.add_argument("npy_file", help="The NPY file containing molecule data")
    parser.add_argument("order_output_file", help="Output filename for the positional encoding data")
    parser.add_argument("resolution", type=int, help="The resolution of the cube (number of cubes per side)")
    return parser.parse_args()

def process_molecule(args):
    index, inchi, xyz_array, chiral_centers, rotation, resolution_dim = args

    # Map one-hot encoded arrays to atom types
    atom_list = []
    atom_coords = []

    for i in range(len(xyz_array)):
        if xyz_array[i][3] == 1:
            atom_list.append('H')
            atom_coords.append(xyz_array[i][:3])
        elif xyz_array[i][4] == 1:
            atom_list.append('C')
            atom_coords.append(xyz_array[i][:3])
        elif xyz_array[i][5] == 1:
            atom_list.append('N')
            atom_coords.append(xyz_array[i][:3])
        elif xyz_array[i][6] == 1:
            atom_list.append('O')
            atom_coords.append(xyz_array[i][:3])
        elif xyz_array[i][7] == 1:
            atom_list.append('F')
            atom_coords.append(xyz_array[i][:3])

    # Convert atom_coords to a numpy array for easier manipulation
    atom_coords = np.array(atom_coords)
    
    # Center atom coordinates around the origin
    atom_coords -= atom_coords.mean(axis=0)

    # Scale coordinates based on the maximum absolute value in any dimension to maintain proportions
    max_abs_coord = np.abs(atom_coords).max()
    if max_abs_coord == 0:
        max_abs_coord = 1  # Avoid division by zero
    atom_coords /= max_abs_coord

    # Scale atom coordinates to fit the new resolution
    scaled_atom_coords = (atom_coords + 1) * ((resolution_dim - 1) / 2)

    # Initialize the heat tensor
    estimated_heat_tensor = np.zeros((resolution_dim, resolution_dim, resolution_dim))

    # Populate the heat tensor and log 1D positions
    agents = []
    for i, coord in enumerate(scaled_atom_coords):
        if len(agents) >= 32:
            break

        # Ensure coordinates are within valid range after scaling
        coord = np.clip(coord, 0, resolution_dim - 1)
        estimated_heat_tensor, applied_positions = agent_heat_component(estimated_heat_tensor, coord[0], coord[1], coord[2], 1, resolution_dim)
        true_x, true_y, true_z = xyz_array[i][:3]
        agent = [atom_list[i], 0, true_x, true_y, true_z] + applied_positions  # Create the agent
        agents.append(agent)

    # Pad agents to ensure it has exactly 32 elements
    null_agent = ["0", 0, -1, -1, -1] + [-1] * 8
    while len(agents) < 32:
        agents.append(null_agent)

    # atom_types = [agent[0] for agent in agents]
    # print(f"Molecule {index} atom types: {' '.join(atom_types)}")

    return {
        'index': index,
        'inchi': inchi,
        'agents': agents,
        'chiral_length': len(chiral_centers)
    }


def construct_tensor_parallel(index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array, resolution_dim):
    pool = Pool(cpu_count())
    args_list = [(index_array[i], inchi_array[i], xyz_arrays[i], chiral_centers_array[i], rotation_array[i], resolution_dim) 
            for i in range(len(index_array))]
    
    results = pool.map(process_molecule, args_list)
    pool.close()
    pool.join()

    # Filter out None results due to errors
    results = [res for res in results if res is not None]

    # Create DataFrame
    df = pd.DataFrame(results)
    return df

def main():
    args = parse_arguments()

    index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array = npy_preprocessor_v4(args.npy_file)
    
    longest_ohe_order = 32
    agent_results = []

    for i in range(len(index_array)):
        molecule_args = (index_array[i], inchi_array[i], xyz_arrays[i], chiral_centers_array[i], rotation_array[i], args.resolution)
        result = process_molecule(molecule_args)
        agent_results.append([result['index'], result['inchi']] + result['agents'][:longest_ohe_order])

    # print(agent_results[0])
    agent_df = pd.DataFrame(agent_results, columns=['index', 'inchi'] + [f'agent{i}' for i in range(32)])
    agent_df.to_csv(args.order_output_file, index=False)

if __name__ == "__main__":
    main()
