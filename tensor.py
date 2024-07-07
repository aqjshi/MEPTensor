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
import periodictable as pt


from wrapper import npy_preprocessor,npy_preprocessor_v2, parse_xyz_string, parse_chiral_centers_string, npy_preprocessor_v2_limit


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
def construct_tensor(molecule_array, weight_array, x_array, y_array, z_array, DENSITY, PRECISION):
    applied_molecules = []
    
    # Round to nearest int 
    cuberoot_density = round(DENSITY ** (1/3))
    tensor = np.zeros((cuberoot_density, cuberoot_density, cuberoot_density))
    # Edge case of empty x array
    if len(x_array) == 0:
        return tensor, applied_molecules
    
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
        x_step_density = (1 / x_scaled_magnitude) * -1
    
    if y_scaled_magnitude > 0:
        y_step_density = y_scaled_magnitude
    elif y_scaled_magnitude < 0:
        y_step_density = (1 / y_scaled_magnitude) * -1

    if z_scaled_magnitude > 0:
        z_step_density = z_scaled_magnitude
    elif z_scaled_magnitude < 0:
        z_step_density = (1 / z_scaled_magnitude) * -1

    for molecule, weight, x, y, z in zip(molecule_array, weight_array, x_array, y_array, z_array): 
        tensor_x = (x - lower_x_bound) / x_scaled_range * x_step_density * cuberoot_density
        tensor_y = (y - lower_y_bound) / y_scaled_range * y_step_density * cuberoot_density
        tensor_z = (z - lower_z_bound) / z_scaled_range * z_step_density * cuberoot_density
    
        applied_molecules.append(molecule)
        append_to_tensor(tensor, tensor_x, tensor_y, tensor_z, weight, PRECISION)
    
    # Round each item in tensor to precision
    tensor = np.round(tensor, PRECISION)
    
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
    # Use nuclear charges
    for molecule in molecule_array:
        weight = 0
        for element in molecule:
            weight += getattr(pt, element).number
        weight_array.append(weight)
    return weight_array

def calculate_repulsion(i, j, k, molecule_array, x_array, y_array, z_array, weight_array):
    repulsion_energy = 0
    R_point = np.array([i, j, k])
    
    for m, (x, y, z, weight) in enumerate(zip(x_array, y_array, z_array, weight_array)):
        R_atom = np.array([x, y, z])
        dist = np.linalg.norm(R_point - R_atom)
        if dist != 0:
            repulsion_energy += weight / dist
            
    return repulsion_energy

#csv master method
def tensor_dataset(index_array, xyz_array, rotation_array, DENSITY, PRECISION):
    tensor_array = []
    formula_array = []
    # some of the xyz data is empty, cannot write by threads, need to do by
    index_str = '\n'.join(map(str, index_array))
    with open('index.txt', 'w') as f:
        f.write(index_str)
    f.close()
    all_data = []
    for i in range(len(index_array)):
        xyz = xyz_array[i]
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
        
        # Calculate repulsion energy here before constraining into the tensor
        repulsion_energy = []
        for x, y, z, weight in zip(x_array, y_array, z_array, weight_array):
            repulsion_energy.append(0.5 * (weight ** 2.4))
        
        tensor, applied_molecules = construct_tensor(molecules, repulsion_energy, x_array, y_array, z_array, DENSITY, PRECISION)
        joined_molecules = ' '.join(applied_molecules)
        tensor_str = ' '.join(' '.join(map(str, row)) for matrix in tensor for row in matrix)
    
        all_data.append([index_array[i], joined_molecules, tensor_str, rotation_array[i][0], rotation_array[i][1], rotation_array[i][2]])
        
        tensor_array.append(tensor)
        formula_array.append(joined_molecules)
        
    with open('small_tensor_dataset.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['index', 'formula', 'tensor', 'rotation0', 'rotation1', 'rotation2'])
        csvwriter.writerows(all_data)

    return tensor_array, formula_array

'''
tensor_dataset_v2.csv 
index, tensor, chiral_length, chiral0, rotation0
'''

#chiral_centers_array is a list of tuples, first item index, second item chiral center type
def tensor_dataset_v2(index_array, xyz_array, chiral_centers_array, rotation_array, DENSITY, PRECISION):
    tensor_array = []
    # some of the xyz data is empty, cannot write by threads, need to do by
    all_data = []
    for i in range(len(index_array)):
    # for i in range(100) :
        xyz = xyz_array[i]
        chiral_center = chiral_centers_array[i]
        
        #
        if len(chiral_center) == 0:
            chiral0 = 0
            chiral_length = 0
        else:
            parsed_chiral_centers = parse_chiral_centers_string(chiral_center)
            chiral0 = parsed_chiral_centers[0]
            chiral_length = len(chiral_center)
        
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
        tensor, applied_molecules = construct_tensor(molecules, weight_array, x_array, y_array, z_array, DENSITY, PRECISION)
        tensor_str = ' '.join(' '.join(map(str, row)) for matrix in tensor for row in matrix)
        all_data.append([index_array[i], tensor_str, chiral_length, chiral0, rotation_array[i][0]])
        tensor_array.append(tensor)
        
    with open('tensor_dataset_v2.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['index', 'tensor', 'chiral_length', 'chiral0', 'rotation0'])
        csvwriter.writerows(all_data)

    return tensor_array
'''
tensor_dataset_v3.csv, only up to 5 chiral centers
index, tensor, chiral0, chiral1, chiral2, chiral3, chiral4, chiral_length, rotation0, rotation1, rotation2
'''

def tensor_dataset_v3(index_array, xyz_array, chiral_centers_array, rotation_array, DENSITY, PRECISION):
    tensor_array = []
    # some of the xyz data is empty, cannot write by threads, need to do by
    all_data = []
    for i in range(len(index_array)):
        xyz = xyz_array[i]
        chiral_center = chiral_centers_array[i]
        
        # edge case of empty chiral center
        if len(chiral_center) == 0:
            chiral0 = 0
            chiral1 = 0
            chiral2 = 0
            chiral3 = 0
            chiral4 = 0
            chiral_length = 0
        else:
            parsed_chiral_centers = parse_chiral_centers_string(chiral_center)
            chiral0 = 0
            chiral1 = 0
            chiral2 = 0
            chiral3 = 0
            chiral4 = 0
            chiral_length = len(chiral_center)
            if len(parsed_chiral_centers) > 0:
                chiral0 = parsed_chiral_centers[0]
            if len(parsed_chiral_centers) > 1:
                chiral1 = parsed_chiral_centers[1]
            if len(parsed_chiral_centers) > 2:
                chiral2 = parsed_chiral_centers[2]
            if len(parsed_chiral_centers) > 3:
                chiral3 = parsed_chiral_centers[3]
            if len(parsed_chiral_centers) > 4:
                chiral4 = parsed_chiral_centers[4]
        
        x_array = []
        y_array = []
        z_array = []
        molecule_array = []

        matrix, molecules = parse_xyz_string(xyz)

        for row, molecule in zip(matrix, molecules):
            x_array.append(row[0])
            y_array.append(row[1])
            z_array.append(row[2])
            
            #turn positive numbers into 1, negative numbers into -1, 0 into 0
            rotation_array[i][0] = 1 if rotation_array[i][0] > 0 else (-1 if rotation_array[i][0] < 0 else 0)
            rotation_array[i][1] = 1 if rotation_array[i][1] > 0 else (-1 if rotation_array[i][1] < 0 else 0)
            rotation_array[i][2] = 1 if rotation_array[i][2] > 0 else (-1 if rotation_array[i][2] < 0 else 0)

            
            molecule_array.append(molecule)
        weight_array = element_into_weight(molecule_array)
        tensor, applied_molecules = construct_tensor(molecules, weight_array, x_array, y_array, z_array, DENSITY, PRECISION)
        tensor_str = ' '.join(' '.join(map(str, row)) for matrix in tensor for row in matrix)
        all_data.append([index_array[i], tensor_str, chiral0, chiral1, chiral2, chiral3, chiral4, chiral_length, rotation_array[i][0], rotation_array[i][1], rotation_array[i][2]])
        tensor_array.append(tensor)
        
    with open('tensor_dataset_v3.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['index', 'tensor', 'chiral0', 'chiral1', 'chiral2', 'chiral3', 'chiral4', 'chiral_length', 'rotation0', 'rotation1', 'rotation2'])
        csvwriter.writerows(all_data)

    return tensor_array



index_array, xyz_array, chiral_centers_array, rotation_array = npy_preprocessor_v2('qm9_filtered.npy')
tensor_array = tensor_dataset_v2(index_array, xyz_array, chiral_centers_array, rotation_array, DENSITY=64, PRECISION=3)



index_array, xyz_array, chiral_centers_array, rotation_array = npy_preprocessor_v2('qm9_filtered.npy')
tensor_array = tensor_dataset_v3(index_array, xyz_array, chiral_centers_array, rotation_array, DENSITY=64, PRECISION=3)




# index_array, xyz_array, rotation_array = npy_preprocessor('qm9_filtered.npy')
# tensor_array, formula_array = tensor_dataset(index_array, xyz_array, rotation_array, DENSITY=64, PRECISION=3)

