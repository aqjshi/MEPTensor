import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive image creation
import matplotlib.pyplot as plt

def decimal_to_bw_image(values, image_size=27):
    """
    Converts an array of decimal values into a black and white image.

    Parameters:
        values (list or np.array): A list or array of 729 decimal values.
        image_size (int): The size of the image (default is 27x27).

    Returns:
        None: Displays the black and white image.
    """
    # Ensure values is a numpy array
    values = np.array(values)

    # Normalize the values between 0 and 1
    normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values)) 

    # Reshape the array into an image_size x image_size matrix
    image_matrix = normalized_values.reshape(image_size, image_size)

    # Display the image
    plt.imshow(image_matrix, cmap='gray', interpolation='nearest')
    plt.axis('off')  # Turn off axis labels
    plt.show()

def decimal_to_rgb_image(values, output_dir='images', filename='output_image.png'):
    """
    Converts an array of 2187 decimal values into a 27x27 RGB image and saves it as a PNG file.
    """
    # Ensure values is a numpy array
    values = np.array(values)

    # Normalize the values between 0 and 1, then scale to [0, 255]
    normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values))
    scaled_values = (normalized_values * 255).astype(np.uint8)  # Scale and convert to 8-bit integer

    # Reshape the array into 729 tuples of 3 values (R, G, B)
    rgb_tuples = scaled_values.reshape(729, 3)

    # Reshape the RGB tuples into a 27x27x3 matrix
    image_matrix = rgb_tuples.reshape(27, 27, 3)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create the full path for the output image
    image_path = os.path.join(output_dir, filename)

    # Save the image as a PNG
    plt.imshow(image_matrix, interpolation='nearest')
    plt.axis('off')  # Turn off axis labels
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0, format='png')
    plt.close()



def read_values_from_file(file_path):
    """
    Reads 2187 decimal values from a text file.

    Parameters:
        file_path (str): Path to the text file.

    Returns:
        np.array: Array of decimal values.
    """
    with open(file_path, 'r') as file:
        values = list(map(float, file.read().strip().split()))
    
    if len(values) != 2187:
        raise ValueError(f"Expected 2187 values but got {len(values)}")
    
    return np.array(values)




def read_voxel_data(file_path):
    """
    Read voxel data from a file and return as a list of numpy arrays.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if not lines:
            return []
        
        return [np.array(list(map(float, line.strip().split()))) for line in lines]


def preprocess_limited_voxels(directory, num_voxels):
    """
    Preprocess a limited number of voxel files in the specified directory.
    """
    files = sorted(os.listdir(directory))[:num_voxels]
    voxel_list = []

    for file in files:
        file_path = os.path.join(directory, file)
        voxel_data = read_voxel_data(file_path)
        
        if voxel_data:
            voxel_list.append(voxel_data[0])  # Use the first voxel in the file
        else:
            print(f"Warning: Skipped file {file_path} due to empty or malformed data.")
    
    return voxel_list


def preprocess_all_voxels(directory):
    """
    Preprocess all voxel files in the specified directory and return combined data.
    """
    combined_voxel_data = []
    files = sorted(os.listdir(directory))

    for file in files:
        file_path = os.path.join(directory, file)
        voxel_data = read_voxel_data(file_path)
        
        if voxel_data:
            combined_voxel_data.extend(voxel_data[0])
        else:
            print(f"Warning: Skipped file {file_path} due to empty or malformed data.")
    
    return np.array(combined_voxel_data)


def preprocess_data(data_dir, num_voxels):
    """
    Preprocess a specified number of voxel data files from the directory.
    """
    print("Starting voxel data preprocessing...")
    voxel_data = preprocess_limited_voxels(data_dir, num_voxels)
    
    # Pad the data to ensure each has 2187 values
    padded_voxel_data = np.array([np.pad(voxel, (0, 2187 - len(voxel)), mode='constant') for voxel in voxel_data])
    
    print(f"Preprocessing complete. Number of voxel images loaded: {len(padded_voxel_data)}")
    return padded_voxel_data


def voxel_to_image(voxel_data):
    """
    Convert voxel data to a 27x27 RGB image.
    """
    r_channel = voxel_data[0::3].reshape((9, 9, 9))
    g_channel = voxel_data[1::3].reshape((9, 9, 9))
    b_channel = voxel_data[2::3].reshape((9, 9, 9))

    # Normalize the channels to [0, 255]
    r_channel = (r_channel * 255).astype(np.uint8)
    g_channel = (g_channel * 255).astype(np.uint8)
    b_channel = (b_channel * 255).astype(np.uint8)

    # Flatten the 3D grid into a 2D 27x27 image
    r_flat = r_channel.flatten().reshape(27, 27)
    g_flat = g_channel.flatten().reshape(27, 27)
    b_flat = b_channel.flatten().reshape(27, 27)

    # Stack the channels to form a 27x27 RGB image
    return np.stack((r_flat, g_flat, b_flat), axis=-1)


def extract_patches(voxel_data, patch_size=9):
    """
    Extract 9x9 patches from a 27x27 voxel image.
    """
    reshaped_data = voxel_data.reshape((27, 27, 3))
    patches = []

    for i in range(0, reshaped_data.shape[0], patch_size):
        for j in range(0, reshaped_data.shape[1], patch_size):
            patch = reshaped_data[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch)
    
    return patches


def visualize_voxel_and_patches(voxel_data, image_title):
    """
    Visualize the voxel image and its patches.
    """
    # Visualize the original voxel image
    image = voxel_to_image(voxel_data)
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(image_title)
    plt.axis('off')
    plt.show()

    # Extract and visualize patches
    patches = extract_patches(voxel_data)
    num_patches = min(10, len(patches))

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for i in range(num_patches):
        ax = axes[i // 5, i % 5]
        ax.imshow(patches[i])
        ax.set_title(f'Patch {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def convert_file_to_image(file_path, output_dir='images'):
    """
    Converts a text file containing 2187 values into a 27x27 black and white image.

    Parameters:
        file_path (str): Path to the text file.
        output_dir (str): Directory to save the image.

    Returns:
        None: Saves the image as a JPG file in the specified directory.
    """
    # Extract the filename without extension to use as the output image name
    filename = os.path.splitext(os.path.basename(file_path))[0] + '.jpg'

    # Read the values from the file
    values = read_values_from_file(file_path)

    # Convert the values to a black and white image and save it
    decimal_to_rgb_image(values, output_dir=output_dir, filename=filename)
                        

def process_and_visualize(data_dir, num_voxels, image_index=0):
    """
    Process and visualize a specific voxel image.
    """
    files = sorted(os.listdir(data_dir))
    
    if image_index >= len(files):
        print(f"Image index {image_index} out of range. Using the first image instead.")
        image_index = 0
    
    selected_file = files[image_index]
    image_title = os.path.splitext(selected_file)[0]

    # Preprocess and visualize the voxel data
    voxel_data = preprocess_data(data_dir, num_voxels)
    selected_voxel = voxel_data[image_index]
    visualize_voxel_and_patches(selected_voxel, image_title)


# # Example usage
if __name__ == "__main__":
    # data_dir = "images/"  # Replace with your data directory
    # num_voxels = 1 # Example number of voxels to preprocess
    # image_index = 0  # Index of the voxel image you want to visualize

    # # Process and visualize the selected voxel image
    # process_and_visualize(data_dir, num_voxels, image_index)
    file_path = "images/005049_0.txt"  # Replace with the path to your file
    output_dir = "final_images"
    convert_file_to_image(file_path, output_dir)