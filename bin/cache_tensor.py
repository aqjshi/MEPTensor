import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from wrapper import npy_preprocessor_v4, heat_component, npy_preprocessor_v4_limit

PRECISION = 3

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

# def decimal_to_bw_image(values, index, chiral_centers, output_dir='images'):
#     """
#     Converts an array of decimal values into a black and white intensity scale, multiplies them by 255, and saves them to a text file.

#     Parameters:
#         values (list or np.array): A list or array of decimal values.
#         index (int): Index to use for the filename.
#         chiral_centers (list): A list of chiral centers to determine the length.
#         output_dir (str): Directory to save the text file.

#     Returns:
#         None: Saves the processed values to a text file.
#     """
#     # Ensure values is a numpy array
#     values = np.array(values)

#     # Normalize the values between 0 and 1
#     normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values))
#     # Multiply each normalized value by 255, repeat each value 3 times, and prepare the string for output
#     output_str = ' '.join(map(str, np.repeat((normalized_values * 255).astype(int), 3)))


#     # Ensure the output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     # Create the filename using the index and the length of chiral_centers
#     posneg = None
#     rs = None
#     filename = f'{index}_{len(chiral_centers)}_{rs}_.txt'

#     # Save the output string to the text file
#     file_path = os.path.join(output_dir, filename)
#     with open(file_path, 'w') as f:
#         f.write(output_str)

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






# def process_molecule_tensor(args):
#     index, inchi, xyz_array, chiral_centers, rotation, resolution_dim = args

#     # Map one-hot encoded arrays to atom types
#     atom_dict = {1: 'H', 2: 'C', 3: 'N', 4: 'O', 5: 'F'}
#     atom_list = []
#     atom_coords = []

#     for i in range(len(xyz_array)):
#         max_idx = np.argmax(xyz_array[i][3:8])
#         if max_idx + 1 in atom_dict:
#             atom_list.append(atom_dict[max_idx + 1])
#             atom_coords.append(xyz_array[i][:3])

#     # Center and scale atom coordinates
#     atom_coords = np.array(atom_coords)
#     atom_coords -= atom_coords.mean(axis=0)

#     max_coord = atom_coords.max(axis=0).max()
#     if max_coord == 0:
#         max_coord = 1  # To avoid division by zero
#     atom_coords /= max_coord

#     # Initialize the heat tensor
#     estimated_heat_tensor = np.zeros((resolution_dim, resolution_dim, resolution_dim))

#     # Scale atom coordinates to fit the new resolution
#     scaled_atom_coords = (atom_coords + 1) * (resolution_dim / 2)

#     # Populate the heat tensor
#     for coord in scaled_atom_coords:
#         heat_component(estimated_heat_tensor, coord[0], coord[1], coord[2], 1, resolution_dim)

#     # Convert tensor values to a 1D list of values for the image
#     tensor_values = [round(estimated_heat_tensor[x, y, z], PRECISION) 
#                      for x in range(resolution_dim) 
#                      for y in range(resolution_dim) 
#                      for z in range(resolution_dim)]

#     # Save the image based on the tensor values
#     denormalize_decimal_to_bw_image(tensor_values, index, chiral_centers)

#     return {
#         'index': index,
#         'inchi': inchi,
#         'tensor': " ".join(map(str, tensor_values)),
#         'chiral_centers': chiral_centers,
#         'chiral_length': len(chiral_centers)
#     }

def process_molecule_tensor(args):
    index, inchi, xyz_array, chiral_centers, posneg, resolution_dim = args

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

    # Convert tensor values to a 1D list of values for the image
    tensor_values = [round(estimated_heat_tensor[x, y, z], PRECISION) 
                     for x in range(resolution_dim) 
                     for y in range(resolution_dim) 
                     for z in range(resolution_dim)]

    # Save the image based on the tensor values and chiral information
    denormalize_decimal_to_bw_image(tensor_values, index, chiral_centers, posneg)

    return {
        'index': index,
        'inchi': inchi,
        'tensor': " ".join(map(str, tensor_values)),
        'chiral_centers': chiral_centers,
        'chiral_length': len(chiral_centers),
        'rotation': posneg
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
    parser.add_argument("output_file_csv", help="The output CSV file for the modified tensor data")
    parser.add_argument("resolution", type=int, help="The resolution of the cube (number of cubes per side)")
    args = parser.parse_args()

    print("init")
    index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array = npy_preprocessor_v4(args.npy_file)
    print("accessed dataset")
    tensor_df= construct_tensor_parallel(index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array, int(args.resolution))
    print("constructed tensor")
    tensor_df.to_csv(args.output_file_csv, index=False)

if __name__ == "__main__":
    main()
