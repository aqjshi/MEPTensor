import numpy as np
import pandas as pd
import re
from scipy.sparse import csr_matrix



def heat_component(tensor, tensor_x, tensor_y, tensor_z, tensor_weight, tensor_dim):
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

    # Add weights to the tensor
    for dx in range(2):
        for dy in range(2):
            for dz in range(2):
                if 0 <= base_x + dx < tensor.shape[0] and 0 <= base_y + dy < tensor.shape[1] and 0 <= base_z + dz < tensor.shape[2]:
                    tensor[base_x + dx, base_y + dy, base_z + dz] += weights[dx, dy, dz]

    return tensor





def csv_preproccessor(filename):
    data = pd.read_csv(filename)
    index_array = data['index']
    molecule_array = data['molecule']
    weight_array = data['weight']
    x_array = data['x']
    y_array = data['y']
    z_array = data['z']
    return index_array, molecule_array, weight_array, x_array, y_array, z_array

def read_data(filename):
    data = np.load(filename, allow_pickle=True)
    df = pd.DataFrame(data.tolist() if data.dtype == 'O' and isinstance(data[0], dict) else data)
    return df

# Function to rewrite npy data into a csv file
def rewrite_npy_to_csv(filename, output_csv):
    df = read_data(filename)
    df.to_csv(output_csv, index=False)


# Function to read the CSV file and return the 'index', 'xyz', and 'rotation' columns
def npy_preprocessor(filename):
    df = read_data(filename)
    return df['index'].values, df['xyz'].values, df['rotation'].values

def npy_preprocessor_inchi(filename):
    df = read_data(filename)
    return df['index'].values, df['inchi'].values

def inchi_to_csv(index_array, inchi_array, output_csv):
    df = pd.DataFrame({'index': index_array, 'inchi': inchi_array})
    df.to_csv(output_csv, index=False)
    return df
# Function to read the CSV file and return the 'index',  'xyz', 'chiral_centers', and 'rotation' columns
def npy_preprocessor_v2(filename):
    df = read_data(filename)
    return df['index'].values, df['xyz'].values, df['chiral_centers'].values, df['rotation'].values

def npy_preprocessor_v2_limit(filename, limit):
    df = read_data(filename)
    return df['index'].values[:limit], df['xyz'].values[:limit], df['chiral_centers'].values[:limit], df['rotation'].values[:limit]

def npy_preprocessor_v3(filename):
    df = read_data(filename)
    return df['index'].values, df['inchi'].values, df['chiral_centers'].values, df['rotation'].values

def npy_preprocessor_v3_limit(filename, limit):
    df = read_data(filename)
    return df['index'].values[:limit], df['inchi'].values[:limit], df['chiral_centers'].values[:limit], df['rotation'].values[:limit]


# Function to parse the matrix string and extract coordinates and elements
def parse_xyz_string(xyz_string):
    xyz_string = str(xyz_string).replace(',', '')
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
    rotation_values = list(map(float, re.findall(r'-?\d+\.\d+', str(rotation_string))))
    return rotation_values

# Function to parse chiral_centers string "[(0, 'S'), (1, 'R'), (2, 'r'), (3, 's'), (4, 'R'), (5, 'S'),(6, 'Tet_CW')]", into a 
def parse_chiral_centers_string(chiral_centers_list):
    # for all tuples, return the second element, to upper
    chiral_centers = [x[1].upper() for x in chiral_centers_list]
    return chiral_centers


#test inchi to csv
index_array, inchi_array = npy_preprocessor_inchi("qm9_filtered.npy")
inchi_to_csv(index_array, inchi_array, "inchi.csv")

# Main execution
# index_array, xyz_array, rotation_array = npy_preprocessor("qm9_filtered.npy", limit = 1000)
# xyz_matrix, molecules = parse_xyz_string(str(xyz_array[0]))

