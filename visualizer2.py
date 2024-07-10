from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def parse_tensor_string(tensor_string, density):
    tensor_values = []
    values = list(map(float, tensor_string.split()))
    idx = 0
    for x in range(density):
        for y in range(density):
            for z in range(density):
                tensor_values.append((x, y, z, values[idx]))
                idx += 1
    return tensor_values

def plot_3d_scatter(tensor_values, density, index):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    xs, ys, zs, values = zip(*tensor_values)
    sc = ax.scatter(xs, ys, zs, c=values, cmap='viridis')

    ax.set_title(f'3D Scatter Plot for Index {index} (Density {density}x{density}x{density})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.colorbar(sc, ax=ax, label='Intensity')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", help="The index of the molecule to visualize", type=int)
    parser.add_argument("density", help="The density of the cube (number of cubes per side)", type=int)
    parser.add_argument("csv_file", help="The CSV file containing tensor data")
    args = parser.parse_args()

    # Load the CSV file
    df = pd.read_csv(args.csv_file)

    # Find the tensor string for the given index
    tensor_string = df[df['index'] == args.index]['tensor_values'].values[0]

    # Parse the tensor string
    tensor_values = parse_tensor_string(tensor_string, args.density)

    # Plot the 3D scatter plot
    plot_3d_scatter(tensor_values, args.density, args.index)

if __name__ == "__main__":
    main()
