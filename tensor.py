# Qingjian Shi 2024



import csv
import sympy as sp
import pandas as pd
import numpy as np
import math
from decimal import Decimal
import numpy as np
import argparse

from helper import parse_xyz_string, npy_preprocessor

#shell args python tensor.py [FILENAME] [DENSITY] [PRECISION]

parser = argparse.ArgumentParser(description="Visualize a 3D tensor as a heat tensor.")
parser.add_argument('filename', type=str, help='The CSV file to process')
parser.add_argument('density', type=int, help='The density of the tensor')
parser.add_argument('precision', type=int, help='The precision for visualization')

args = parser.parse_args()

FILENAME = args.filename
DENSITY = args.density
PRECISION = args.precision


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
def construct_tensor(molecule_array, weight_array, x_array, y_array, z_array, DENSITY):
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
def tensor_dataset(index_array, xyz_array, rotation_array):
    #write the indexes first
    with open('index.txt', 'w') as f:
        for index in index_array:
            f.write(str(index) + '\n')
        # print(index_array) #output "['079782' '005049']"
        f.close()
    #write the rotations
    with open('rotation.txt', 'w') as f:
        for rotation in rotation_array:
            f.write(str(rotation) + '\n')
        # print(rotation_array) #output "[list([-0.0, -0.0, -0.0]) list([-14.99, -12.5, -4979.91])]""
        f.close()
    #write the tensors
    tensor_array = []
    formula_array = []
    for xyz in zip(xyz_array):
        x_array = []
        y_array = []
        z_array = []
        ##2d array of strings
        molecule_array = []
        
        matrix, molecules = parse_xyz_string(xyz)
        
        for row, molecule in zip(matrix, molecules):
            x_array.append(row[0])
            y_array.append(row[1])
            z_array.append(row[2])
            molecule_array.append(molecule)
        weight_array = element_into_weight(molecule_array)  
        tensor, applied_molecules = construct_tensor(molecules, weight_array, x_array, y_array, z_array, DENSITY)
        print("hello")
        print(applied_molecules)
        joined_molecules = ' '.join(applied_molecules)
        print(joined_molecules)
        tensor_array.append(tensor)
        formula_array.append(joined_molecules)
    #write the tensor
    with open('tensor.txt', 'w') as f:
        for tensor, formula in zip(tensor_array, formula_array):
            f.write("$tensor\n")
            f.write(str(tensor) + '\n')
            f.write("^molecules\n")
            f.write(str(formula) + '\n')
        # print(tensor_array) 
        '''
        SAMPLE OUTPUT
        array([[[0.04, 0.  , 0.  , 0.  ],
        [1.3 , 1.38, 0.  , 0.  ],
        [0.46, 0.74, 0.  , 0.  ],
        [0.15, 0.02, 0.  , 0.  ]],

       [[0.  , 0.  , 0.82, 1.23],
        [1.23, 4.99, 1.46, 2.06],
        [8.69, 3.57, 0.03, 0.  ],
        [4.03, 0.85, 0.19, 0.64]],

       [[0.  , 0.  , 1.67, 2.36],
        [0.  , 0.  , 3.15, 4.43],
        [0.13, 1.29, 0.49, 0.  ],
        [0.09, 6.98, 2.95, 3.58]],

       [[0.  , 0.  , 1.22, 1.71],
        [0.84, 4.51, 2.49, 3.49],
        [7.11, 4.89, 0.  , 0.  ],
        [4.51, 1.11, 0.  , 0.01]]])]
        '''
        # print(formula_array) #output ['O C C C O C C N C H H H H H H H', 'N C N C C N C N H H H H']
        f.close()
    return tensor_array, formula_array
    

# index_array, molecule_array, weight_array, x_array, y_array, z_array = csv_preproccessor(FILENAME)
#import from another python file called helper.py



#npy_preproccessor returns index_array, xyz_array, rotation_array
index_array, xyz_array, rotation_array = npy_preprocessor(FILENAME)

tensor_array, formula_array = tensor_dataset(index_array, xyz_array, rotation_array)


# tensor, applied_molecules = construct_tensor(index_array, molecule_array, weight_array, x_array, y_array, z_array, DENSITY)

# print (tensor)
# print (applied_molecules)


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Assuming tensor is a 3D numpy array
# x, y, z = np.indices(tensor.shape)

# # Normalize tensor values for coloring
# norm_tensor = tensor / tensor.max()

# # Plot
# ax.scatter(x, y, z, c=norm_tensor.flatten(), cmap='viridis')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()

# #close plt
# plt.close()
