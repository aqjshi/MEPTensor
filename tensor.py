# Qingjian Shi 2024



import csv
import sympy as sp
import pandas as pd
import numpy as np
import math
from decimal import Decimal
import numpy as np
import argparse
from scipy.sparse import csr_matrix

from helper import parse_xyz_string, npy_preprocessor

#shell args python tensor.py [FILENAME] [DENSITY] [PRECISION]

# parser = argparse.ArgumentParser(description="Visualize a 3D tensor as a heat tensor.")
# parser.add_argument('filename', type=str, help='The CSV file to process')
# parser.add_argument('density', type=int, help='The density of the tensor')
# parser.add_argument('precision', type=int, help='The precision for visualization')

# args = parser.parse_args()

# FILENAME = args.filename
# DENSITY = args.density
# PRECISION = args.precision

#if null param
FILENAME = "qm9_filtered.npy"
DENSITY = 64
PRECISION = 2


#scale number into the highest precision, return the number, the magnitude, and the precision
def scale_to_integer(num):
    if num == 0:
        return 0, 1
    d = Decimal(str(num))
    # Handle large numbers
    if abs(num) >= 100:
        true_magnitude = math.floor(math.log10(abs(num)))
        scaled_num = round(num / 10**(true_magnitude - 1))
        return scaled_num, -10**(true_magnitude - 1)

    # Handle small numbers
    elif abs(num) < 10:
        true_magnitude = math.floor(math.log10(abs(num)))
        scaled_num = round(num * 10**abs(true_magnitude) * 10) 
        return scaled_num, 10**-(true_magnitude - 1)

    # Handle numbers in between
    else:
        scaled_num = round(num, 1)
        return scaled_num, 1


#helper method to safe add distributed heat tensor to a tensor
def heat_component(tensor_x, tensor_y, tensor_z, tensor_weight, PRECISION):
    # New tensor 2x2x2
    tensor = np.zeros((2, 2, 2))

    # Get the decimal part for bleeding
    bleed_x = tensor_x % 1
    bleed_y = tensor_y % 1
    bleed_z = tensor_z % 1
    
    complement_bleed_x = 1 - bleed_x
    complement_bleed_y = 1 - bleed_y
    complement_bleed_z = 1 - bleed_z    
    

    # Calculate the weights for each of the neighboring cells
    tensor[0, 0, 0] = complement_bleed_x * complement_bleed_y * complement_bleed_z * tensor_weight
    tensor[0, 0, 1] = complement_bleed_x * complement_bleed_y * bleed_z * tensor_weight
    tensor[0, 1, 0] = complement_bleed_x * bleed_y * complement_bleed_z * tensor_weight
    tensor[0, 1, 1] = complement_bleed_x * bleed_y * bleed_z * tensor_weight
    tensor[1, 0, 0] = bleed_x * complement_bleed_y * complement_bleed_z * tensor_weight
    tensor[1, 0, 1] = bleed_x * complement_bleed_y * bleed_z * tensor_weight
    tensor[1, 1, 0] = bleed_x * bleed_y * complement_bleed_z * tensor_weight
    tensor[1, 1, 1] = bleed_x * bleed_y * bleed_z * tensor_weight
    
    

    return tensor

#testing the heat component
# print(heat_component(0.34, 0.5, 0.5, 1, 2))


# #version 1

#["index", "molecule", "weight", "x", "y", "z"]

#preprocess the csv file into processed arrays
def csv_preproccessor(filename):
    data = pd.read_csv(filename)
    index_array = data['index']
    molecule_array = data['molecule']
    weight_array = data['weight']
    x_array = data['x']
    y_array = data['y']
    z_array = data['z']
    return index_array, molecule_array, weight_array, x_array, y_array, z_array

#constructs empty
def construct_tensor(molecule_array, weight_array, x_array, y_array, z_array, DENSITY):
    applied_molecules = []
    
    #round to nearest int 
    cuberoot_density = round(DENSITY ** (1/3))
    tensor = np.zeros((cuberoot_density, cuberoot_density, cuberoot_density))
    
    
    upper_x_bound = max(x_array)
    lower_x_bound = min(x_array)

    upper_y_bound = max(y_array)
    lower_y_bound = min(y_array)
    
    upper_z_bound = max(z_array)
    lower_z_bound = min(z_array)
    

    x_scaled_range, x_scaled_magnitude = scale_to_integer((upper_x_bound - lower_x_bound))
    y_scaled_range, y_scaled_magnitude = scale_to_integer((upper_y_bound - lower_y_bound))
    z_scaled_range, z_scaled_magnitude = scale_to_integer((upper_z_bound - lower_z_bound))
    
    x_step_density = 1
    y_step_density = 1
    z_step_density = 1
    
    if x_scaled_magnitude > 0:
        x_step_density = x_scaled_magnitude
    elif x_scaled_magnitude < 0:
        x_step_density = (1/x_scaled_magnitude) * -1
    
    if y_scaled_magnitude  > 0:
        y_step_density = y_scaled_magnitude
    elif y_scaled_magnitude  < 0:
        y_step_density = (1/y_scaled_magnitude) * -1

    if z_scaled_magnitude  > 0:
        z_step_density = z_scaled_magnitude
    elif z_scaled_magnitude  < 0:
        z_step_density = (1/z_scaled_magnitude) * -1
        
    for molecule, weight, x, y, z in zip(molecule_array, weight_array, x_array, y_array, z_array): 
        tensor_x = (x - lower_x_bound) / x_scaled_range * x_step_density * cuberoot_density
        tensor_y = (y - lower_y_bound) / y_scaled_range * y_step_density * cuberoot_density
        tensor_z = (z - lower_z_bound) / z_scaled_range * z_step_density * cuberoot_density
    
        applied_molecules.append(molecule)
        append_to_tensor(tensor, tensor_x, tensor_y, tensor_z, weight, PRECISION)

    return tensor, applied_molecules

def append_to_tensor(tensor, tensor_x, tensor_y, tensor_z, tensor_weight, PRECISION):
    heat = heat_component(tensor_x, tensor_y, tensor_z, tensor_weight, PRECISION)
    
    starting_x = int(tensor_x)
    starting_y = int(tensor_y)
    starting_z = int(tensor_z)
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                if starting_x + i < tensor.shape[0] and starting_y + j < tensor.shape[1] and starting_z + k < tensor.shape[2]:
                    tensor[starting_x + i, starting_y + j, starting_z + k] += round(heat[i, j, k], PRECISION)
    
    return tensor
 

def element_into_weight(molecule_array):
    weight_array = []
    for molecule in molecule_array:
        if molecule == 'H':
            weight_array.append(1)
        elif molecule == 'C':
            weight_array.append(12)
        elif molecule == 'N':
            weight_array.append(14)
        elif molecule == 'O':
            weight_array.append(16)
        elif molecule == 'F':
            weight_array.append(19)
        elif molecule == 'Cl':
            weight_array.append(35.5)
        elif molecule == 'Br':
            weight_array.append(80)
        elif molecule == 'I':
            weight_array.append(127)
    return weight_array

#tensor dataset is going to output into 3 different temp txt files, index.txt, tensor.txt, and rotation.txt
#returns index_array, tensor_array, formula_array, rotation_array
# def tensor_dataset(index_array, xyz_array, rotation_array):
#     #write the indexes first
#     with open('index.txt', 'w') as f:
#         for index in index_array:
#             f.write(str(index) + '\n')
#         # print(index_array) #output "079782 005049"
#         f.close()
#     #write the rotations
#     with open('rotation.txt', 'w') as f:
#         for rotation in rotation_array:
#             #write into file from join the array into a string separated by space
#             for i in range(len(rotation)):
#                 f.write(str(rotation[i]) + ' ')
#             f.write('\n')
#             # f.write(str(rotation) + '\n')
#         # print(rotation_array) #output "-0.0, -0.0, -0.0"
#         f.close()
#     #write the tensors
#     tensor_array = []
#     formula_array = []
#     for xyz in zip(xyz_array):
#         x_array = []
#         y_array = []
#         z_array = []
#         ##2d array of strings
#         molecule_array = []
        
#         matrix, molecules = parse_xyz_string(xyz)
        
#         for row, molecule in zip(matrix, molecules):
#             x_array.append(row[0])
#             y_array.append(row[1])
#             z_array.append(row[2])
#             molecule_array.append(molecule)
#         weight_array = element_into_weight(molecule_array)  
#         tensor, applied_molecules = construct_tensor(molecules, weight_array, x_array, y_array, z_array, DENSITY)
#         joined_molecules = ' '.join(applied_molecules)
#         tensor_array.append(tensor)
#         formula_array.append(joined_molecules)
#     #write the tensor
#     with open('tensor.txt', 'w') as f:
#         for tensor in tensor_array:
#             for matrix in tensor:
#                 for row in matrix:
#                     f.write(' '.join(map(str, row))) #output "0.0 0.0 0.0 0.050.0 0.01 0.08 0.00.0 0.36 1.49 0.00.0 0.05 0.13 0.00.0 0.0 0.59 2.01.02 2.0 2.37 3.630.21 2.18 8.92 0.010.0 0.0 3.2 4.770.0 2.68 6.42 0.91.1099999999999999 5.35 2.74 3.250.63 3.46 0.01 7.380.02 0.12 3.2199999999999998 4.790.0 0.71 1.59 0.00.17 0.22 7.02 0.83000000000000011.72 8.290000000000001 3.4 2.871.97 6.159999999999999 0.01 0.0"
#             f.write('\n')
#             # f.write(str(tensor) + '\n')
#         # print(tensor_array) 
#         f.close()
#     with open('formula.txt', 'w') as f:
#         for formula in formula_array:
#             f.write(str(formula) + '\n') #output "O C C C O C C N C H H H H H H H"
#         f.close()
    
#     return tensor_array, formula_array
    
#csv master method
def tensor_dataset(index_array, xyz_array, rotation_array):
    tensor_array = []
    formula_array = []

    for xyz in xyz_array:
        x_array = []
        y_array = []
        z_array = []
        molecule_array = []

        matrix, molecules = parse_xyz_string(xyz)

        for row, molecule in zip(matrix, molecules):
            x_array.append(row[0])
            y_array.append(row[1])
            z_array.append(row[2])
            molecule_array.append(molecule)

        weight_array = element_into_weight(molecule_array)
        tensor, applied_molecules = construct_tensor(molecules, weight_array, x_array, y_array, z_array, DENSITY)
        joined_molecules = ' '.join(applied_molecules)
        tensor_array.append(tensor)
        formula_array.append(joined_molecules)

    with open('tensor_dataset.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['index', 'formula', 'tensor', 'rotation0', 'rotation1', 'rotation2'])

        for i in range(len(index_array)):
            index = index_array[i]
            formula = formula_array[i]
            tensor = ' '.join(' '.join(map(str, row)) for matrix in tensor_array[i] for row in matrix)
            rotation0 = rotation_array[i][0]
            rotation1 = rotation_array[i][1]
            rotation2 = rotation_array[i][2]
            csvwriter.writerow([index, formula, tensor, rotation0, rotation1, rotation2])

    return tensor_array, formula_array

#npy_preproccessor returns index_array, xyz_array, rotation_array
index_array, xyz_array, rotation_array = npy_preprocessor(FILENAME)

tensor_array, formula_array = tensor_dataset(index_array, xyz_array, rotation_array)

#train_X is tensor_array, formula_array, train_Y is rotation_array

