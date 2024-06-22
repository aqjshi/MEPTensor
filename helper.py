import numpy as np
import pandas as pd
import re
from scipy.sparse import csr_matrix
# Function to load and inspect the .npy file and take the first 10 items
def read_data(filename):
    data = np.load(filename, allow_pickle=True)
    if data.dtype == 'O' and isinstance(data[0], dict):
        df = pd.DataFrame(data.tolist())
    else:
        df = pd.DataFrame(data)
    # Take the first 1000 items
    df = df.head(1000)
    
    return df

# Function to rewrite npy data into a csv file
def rewrite_npy_to_csv(filename, output_csv):
    df = read_data(filename)
    df.to_csv(output_csv, index=False)

# Example usage
# rewrite_npy_to_csv("qm9_filtered.npy", "qm9_filtered.csv")

# Function to read the CSV file and return the 'xyz' column's first value
def npy_preprocessor(filename):
    data = np.load(filename, allow_pickle=True)
    if data.dtype == 'O' and isinstance(data[0], dict):
        df = pd.DataFrame(data.tolist())
    else:
        df = pd.DataFrame(data)
    # Take the first 2 items
    df = df.head(2)
    index_array, xyz_array, rotation_array = df['index'].values, df['xyz'].values, df['rotation'].values
    return index_array, xyz_array, rotation_array

# Function to parse the matrix string and extract coordinates and elements
def parse_xyz_string(xyz_string):
    
    xyz_string = str(xyz_string)
    #clean string of ,
    xyz_string = xyz_string.replace(',', '')
    
    elements = ['H', 'C', 'N', 'O', 'F']
    rows = re.findall(r'\[([^\[\]]*)\]', xyz_string)

    xyz_matrix = []
    molecules = []
    
    for row in rows:
        numbers = list(map(float, row.split()))
        if all(numbers[:3]):  # Check if the first three items are non-zero
            xyz_matrix.append(numbers[:3])
            for i in range(3, 8):
                if numbers[i] == 1:
                    molecules.append(elements[i-3])
                    break
    
    return xyz_matrix, molecules

# Function to parse the rotation string into an array of floats
def parse_rotation_string(rotation_string):
    rotation_string =  str(rotation_string)
    # Strip the string of brackets and spaces, then split by commas
    rotation_values = list(map(float, re.findall(r'-?\d+\.\d+', rotation_string)))
    return rotation_values

# Function to write the matrix to a text file
def write_matrix_to_txt(matrix, filename):
    with open(filename, 'w') as f:
        for row in matrix:
            f.write(' '.join(map(str, row)) + '\n')


# Main execution
# index_array, xyz_array, rotation_array = npy_preprocessor("qm9_filtered.npy")
# xyz_matrix, molecules = parse_xyz_string(str(xyz_array[0]))
# write_matrix_to_txt(xyz_matrix, "matrix.txt")
