#3d tensor generation of the molecules

#shell args  python tensor.py filename density precision


# DENSITY = args[3]
# PRECISION = args[4]


DENSITY = 64
PRECISION = 2

import csv
import sympy as sp
import pandas as pd
import numpy as np
import math
from decimal import Decimal
import numpy as np

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
print(heat_component(0.34, 0.5, 0.5, 1, 2))


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
def construct_tensor(index_array, molecule_array, weight_array, x_array, y_array, z_array, DENSITY):
    applied_molecules = []
    
    #round to nearest int 
    cuberoot_density = round(DENSITY ** (1/3))
    print(cuberoot_density)
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
    
    top_left_x = int(tensor_x)
    top_left_y = int(tensor_y)
    top_left_z = int(tensor_z)
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                if top_left_x + i < tensor.shape[0] and top_left_y + j < tensor.shape[1] and top_left_z + k < tensor.shape[2]:
                    tensor[top_left_x + i, top_left_y + j, top_left_z + k] += round(heat[i, j, k], PRECISION)
    
    return tensor
 
filename = 'xyz.csv' 
index_array, molecule_array, weight_array, x_array, y_array, z_array = csv_preproccessor(filename)
tensor, applied_molecules = construct_tensor(index_array, molecule_array, weight_array, x_array, y_array, z_array, DENSITY)

print (tensor)
print (applied_molecules)
