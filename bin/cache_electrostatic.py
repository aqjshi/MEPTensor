import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from wrapper import npy_preprocessor_v4, heat_component, npy_preprocessor_v4_limit

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
    index, inchi, xyz_array, chiral_centers, posneg, resolution_dim = args

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

    tensor_values = calculate_electrostatic_potential_energy(tensor_positions, scaled_atom_coords, nuclear_charges)
    combined_tensor_values = [f"{round(val, PRECISION)}" for val in tensor_values]

    denormalize_decimal_to_bw_image(tensor_values, index, chiral_centers, posneg)

    return {
        'index': index,
        'inchi': inchi,
        'tensor': " ".join(map(str, tensor_values)),
        'chiral_centers': chiral_centers,
        'chiral_length': len(chiral_centers),
        'rotation': posneg
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

def decimal_to_bw_image(values, index, output_dir='images'):
    """
    Converts an array of decimal values into a black and white image and saves it.

    Parameters:
        values (list or np.array): A list or array of decimal values.
        index (int): Index to use for the filename.
        output_dir (str): Directory to save the image.

    Returns:
        None: Saves the image to the specified directory.
    """
    # Determine the appropriate image size
    image_size = int(np.sqrt(len(values)))

    # Ensure values is a numpy array
    values = np.array(values)

    # Normalize the values between 0 and 1
    normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values))

    # Reshape the array into an image_size x image_size matrix
    image_matrix = normalized_values.reshape(image_size, image_size)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the image
    image_path = os.path.join(output_dir, f'{index}.jpg')
    # plt.imshow(image_matrix, cmap='gray', interpolation='nearest')
    plt.axis('off')  # Turn off axis labels
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def denormalize_decimal_to_bw_image(values, index, chiral_centers, posneg, output_dir='final_images'):
    """
    Converts an array of decimal values into a black and white intensity scale, multiplies them by 255, and saves them to a text file.

    Parameters:
        values (list or np.array): A list or array of decimal values.
        index (int): Index to use for the filename.
        chiral_centers (list): A list of chiral centers to determine the length.
        posneg (str): A string indicating whether the molecule is positive or negative.
        output_dir (str): Directory to save the text file.

    Returns:
        None: Saves the processed values to a text file.
    """
    # Ensure values is a numpy array
    values = np.array(values)

    # Multiply each normalized value by 255, repeat each value 3 times, and prepare the string for output
    output_str = ' '.join(map(str, np.repeat((values * 255).astype(int), 3)))

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get the R/S value and the length of chiral centers
    chiral_length = len(chiral_centers)
    rs = chiral_centers[0] if chiral_length > 0 else '0'

    # Create the filename using the index, chiral length, rs, and posneg
    filename = f'{index}_chiralLength{chiral_length}_{rs}_{posneg}.txt'

    # Save the output string to the text file
    file_path = os.path.join(output_dir, filename)
    with open(file_path, 'w') as f:
        f.write(output_str)




def denormalize_decimal_to_bw_image(values, index, chiral_centers, posneg, output_dir='images'):
    """
    Converts an array of decimal values into a black and white intensity scale, multiplies them by 255, and saves them to a text file.

    Parameters:
        values (list or np.array): A list or array of decimal values.
        index (int): Index to use for the filename.
        chiral_centers (list): A list of chiral centers to determine the length.
        posneg (str): A string indicating whether the molecule is positive or negative.
        output_dir (str): Directory to save the text file.

    Returns:
        None: Saves the processed values to a text file.
    """
    # Ensure values is a numpy array
    values = np.array(values)

    # Multiply each normalized value by 255, repeat each value 3 times, and prepare the string for output
    output_str = ' '.join(map(str, np.repeat((values * 255).astype(int), 3)))

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get the R/S value and the length of chiral centers
    chiral_length = len(chiral_centers)
    rs = chiral_centers[0] if chiral_length > 0 else '0'
    # if 'R' or 'r' then cast to upper, if 'S' or 's' then cast to upper, else keep the same
    #null case exception
    if chiral_length == 0:
        casted_rs = '0'
    else:
        casted_rs = 'R' if rs[1][0].lower() == 'r' else 'S' if rs[1][0].lower() == 's' else rs[1]

    # Create the filename using the index, chiral length, rs, and posneg
    filename = f'{index}${chiral_length}${casted_rs}${0 if posneg[0] == 0 else 1 if posneg[0] >= 0 else -1}.txt'

    # Save the output string to the text file
    file_path = os.path.join(output_dir, filename)
    with open(file_path, 'w') as f:
        f.write(output_str)





def process_molecule_order(resolution_dim):
    order_dim = resolution_dim**3
    tensor_values = " ".join(str(i + 1) for i in range(order_dim))  # values from 1 to order_dim
    return {'tensor': tensor_values}

def construct_tensor_parallel(index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array, resolution_dim):
    pool = Pool(cpu_count())
    args = [(index_array[i], inchi_array[i], xyz_arrays[i], chiral_centers_array[i], rotation_array[i], resolution_dim) 
            for i in range(len(index_array))]
    
    tensor_results = pool.map(process_molecule, args)
    
    pool.close()
    pool.join()

    # Filter out None results due to errors
    tensor_results = [res for res in tensor_results if res is not None]

    # Create DataFrames
    tensor_df = pd.DataFrame(tensor_results)
    
    return tensor_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_file", help="The NPY file containing molecule data")
    args = parser.parse_args()

    print("init")
    index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array = npy_preprocessor_v4(args.npy_file)
    print("accessed dataset")
    tensor_df= construct_tensor_parallel(index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array, 16)
    print("constructed tensor")

if __name__ == "__main__":
    main()
