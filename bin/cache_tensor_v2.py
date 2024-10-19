import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import re
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count
from wrapper import npy_preprocessor_v4, heat_component

PRECISION = 3

def denormalize_decimal_to_bw_image(values, index, chiral_centers, posneg, rotated, output_dir='images'):
    """
    Converts an array of decimal values into a black and white intensity scale, multiplies them by 255, and saves them to a text file.
    """
    # only run function for molecules with less than 2 chiral centers
    if len(chiral_centers) > 1 or len(chiral_centers) == 0:
        return
    values = np.array(values)
    output_str = ' '.join(map(str, np.repeat((values * 255).astype(int), 3)))
    os.makedirs(output_dir, exist_ok=True)
    
    chiral_length = len(chiral_centers)
    rs = chiral_centers[0] if chiral_length > 0 else '0'
    # chiral_length  > 0
    if chiral_length == 1:
        casted_rs = 'R' if re.search(r'\bR\b|\bTet_CW\b', rs[1], re.IGNORECASE) else 'S' if re.search(r'\bS\b|\bTet_CCW\b', rs[1], re.IGNORECASE) else rs[1]
    # elif chiral_length == 0:
    #     casted_rs = '0'
    filename = f'{index}${chiral_length}${casted_rs}${1 if posneg[0]> 0 else 0}${rotated}.txt'
    file_path = os.path.join(output_dir, filename)
    
    with open(file_path, 'w') as f:
        f.write(output_str)

def plot_structure(ax, points, label, color='b'):
    points = np.array(points)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, label=label)
    for point in points:
        ax.text(point[0], point[1], point[2], f'({point[0]:.1f},{point[1]:.1f},{point[2]:.1f})', color=color)

def rotate_structure(points, Rx, Ry, Rz):
    rotated_points = []
    for point in points:
        rotated_point = Rz @ (Ry @ (Rx @ point))  # Apply rotations in sequence X -> Y -> Z
        rotated_points.append(rotated_point)
    return rotated_points

def process_molecule_tensor(index, xyz_array, chiral_centers, posneg, resolution_dim, rotated, output_dir):
    # Center and scale atom coordinates
    atom_coords = xyz_array[:, :3]
    atom_coords -= atom_coords.mean(axis=0)
    max_coord = atom_coords.max(axis=0).max() or 1  # Avoid division by zero
    atom_coords /= max_coord

    # Initialize heat tensor
    estimated_heat_tensor = np.zeros((resolution_dim, resolution_dim, resolution_dim))
    scaled_atom_coords = (atom_coords + 1) * (resolution_dim / 2)
    
    for coord in scaled_atom_coords:
        heat_component(estimated_heat_tensor, coord[0], coord[1], coord[2], 1, resolution_dim)

    tensor_values = [round(estimated_heat_tensor[x, y, z], PRECISION) 
                     for x in range(resolution_dim) 
                     for y in range(resolution_dim) 
                     for z in range(resolution_dim)]
    
    denormalize_decimal_to_bw_image(tensor_values, index, chiral_centers, posneg, rotated, output_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_file", help="The NPY file containing molecule data")
    parser.add_argument("resolution", type=int, help="The resolution of the cube (number of cubes per side)")
    args = parser.parse_args()

    # Load dataset
    print("Loading dataset")
    index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array = npy_preprocessor_v4(args.npy_file)

    # Filter only molecules with 0 or 1 chiral centers
    filtered_indices = [i for i in range(len(index_array)) if len(chiral_centers_array[i]) in [0, 1]]
    filtered_index_array = [index_array[i] for i in filtered_indices]
    filtered_xyz_arrays = [xyz_arrays[i] for i in filtered_indices]
    filtered_chiral_centers_array = [chiral_centers_array[i] for i in filtered_indices]
    filtered_rotation_array = [rotation_array[i] for i in filtered_indices]

    # Split data into train and test sets
    train_indices, test_indices = train_test_split(range(len(filtered_index_array)), test_size=0.2, random_state=42, shuffle=True)

    # Create directories for train and test
    os.makedirs('train', exist_ok=True)
    os.makedirs('test', exist_ok=True)

    resolution_dim = args.resolution

    # Process test set (no rotation)
    for i in test_indices:
        index = filtered_index_array[i]
        xyz_array = filtered_xyz_arrays[i]
        chiral_centers = filtered_chiral_centers_array[i]
        posneg = filtered_rotation_array[i]
        process_molecule_tensor(index, xyz_array, chiral_centers, posneg, resolution_dim, rotated=0, output_dir='test_rs')

    # Process train set (original and rotated)
    for i in train_indices:
        index = filtered_index_array[i]
        xyz_array = filtered_xyz_arrays[i]
        chiral_centers = filtered_chiral_centers_array[i]
        posneg = filtered_rotation_array[i]

        # Original molecule
        process_molecule_tensor(index, xyz_array, chiral_centers, posneg, resolution_dim, rotated=0, output_dir='train_rs')

        # Rotated molecule
       # Apply 3 different rotations to each molecule
        rotation_angles = [(30, 45, 60)]

        for rotation_idx, (theta_x_deg, theta_y_deg, theta_z_deg) in enumerate(rotation_angles):
            theta_x, theta_y, theta_z = np.radians(theta_x_deg), np.radians(theta_y_deg), np.radians(theta_z_deg)
            
            # Rotation matrices
            Rx = np.array([[1, 0, 0],
                        [0, np.cos(theta_x), -np.sin(theta_x)],
                        [0, np.sin(theta_x), np.cos(theta_x)]])
            Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                        [0, 1, 0],
                        [-np.sin(theta_y), 0, np.cos(theta_y)]])
            Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                        [np.sin(theta_z), np.cos(theta_z), 0],
                        [0, 0, 1]])
            
            rotated_xyz_array = np.array(rotate_structure(xyz_array[:, :3], Rx, Ry, Rz))
            
            # Process each rotated molecule
            process_molecule_tensor(index, rotated_xyz_array, chiral_centers, posneg, resolution_dim, rotated=rotation_idx + 1, output_dir='train_rs')


if __name__ == "__main__":
    main()
