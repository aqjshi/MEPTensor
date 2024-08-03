import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import argparse
from wrapper import heat_component, npy_preprocessor

def tensor_parser(tensor_string, density):
    values = list(map(float, tensor_string.split()))
    return [(x, y, z, values[idx]) for idx, (x, y, z) in enumerate(np.ndindex(density, density, density))]

def add_heat_to_axes(ax, tensor_values):
    x, y, z, v = zip(*tensor_values)
    sc = ax.scatter(x, y, z, c=v, cmap='coolwarm', alpha=0.5)
    plt.colorbar(sc, ax=ax, orientation='horizontal')
    return ax

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="The CSV file containing InChI data")
    parser.add_argument("npy_file", help="The NPY file containing molecule data")
    parser.add_argument("index", help="The index of the molecule to visualize", type=int)
    parser.add_argument("density", help="The density of the cube (number of cubes per side)", type=int)
    args = parser.parse_args()

    inchi_df = pd.read_csv(args.filename)
    print("Available columns in InChI dataset:", inchi_df.columns)

    index = args.index
    inchi_string = inchi_df.loc[inchi_df['index'] == index, 'inchi'].values[0]
    tensor_string = inchi_df.loc[inchi_df['index'] == index, 'tensor'].values[0]

    molecule = Chem.MolFromInchi(inchi_string)
    if molecule is None:
        print(f"Invalid InChI string for index {index}")
        return

    molecule = Chem.AddHs(molecule)
    if not molecule.GetNumConformers():
        AllChem.EmbedMolecule(molecule, randomSeed=42)
        AllChem.UFFOptimizeMolecule(molecule)

    img_2d = Draw.MolToImage(molecule, size=(300, 300))

    conf = molecule.GetConformer()
    atom_coords = conf.GetPositions()
    atom_types = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    atom_coords = (atom_coords - atom_coords.mean(axis=0)) / atom_coords.max(axis=0).max()

    fig = plt.figure(figsize=(60, 10))
    ax1 = fig.add_subplot(161)
    ax1.imshow(img_2d)
    ax1.set_title(f'2D {index}')
    ax1.axis('off')

    ax2 = fig.add_subplot(162, projection='3d')
    ax2.scatter(atom_coords[:, 0], atom_coords[:, 1], atom_coords[:, 2], c='b', s=100)
    for i, atom_type in enumerate(atom_types):
        ax2.text(atom_coords[i, 0], atom_coords[i, 1], atom_coords[i, 2], atom_type, fontsize=12, color='r')
    ax2.set_title('Build from RDKit')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.grid(False)

    npy_data = np.load(args.npy_file, allow_pickle=True)
    npy_df = pd.DataFrame(npy_data.tolist() if npy_data.dtype == 'O' and isinstance(npy_data[0], dict) else npy_data)
    print("First few rows of npy_df:")
    print(npy_df.head())

    index_array, xyz_arrays, rotation_arrays = npy_preprocessor(args.npy_file)
    atom_list, atom_coords_npy = [], []

    atom_dict = {1: 'H', 2: 'C', 3: 'N', 4: 'O', 5: 'F'}
    for i, idx in enumerate(index_array):
        if int(idx) == int(args.index):
            for xyz_array in xyz_arrays[i]:
                max_idx = np.argmax(xyz_array[3:8])
                if max_idx + 1 in atom_dict:
                    atom_list.append(atom_dict[max_idx + 1])
                    atom_coords_npy.append(xyz_array[:3])

    atom_coords_npy = np.array(atom_coords_npy)
    atom_coords_npy = (atom_coords_npy - atom_coords_npy.mean(axis=0)) / (atom_coords_npy.max(axis=0) - atom_coords_npy.min(axis=0)).max()

    ax3 = fig.add_subplot(163, projection='3d')
    for i, atom_type in enumerate(atom_list):
        ax3.scatter(atom_coords_npy[i, 0], atom_coords_npy[i, 1], atom_coords_npy[i, 2], c='b', s=100)
        ax3.text(atom_coords_npy[i, 0], atom_coords_npy[i, 1], atom_coords_npy[i, 2], atom_type, fontsize=12, color='r')
    ax3.set_title('Build from NPY file')

    raw_tensor_values = tensor_parser(tensor_string, args.density)
    raw_x, raw_y, raw_z, raw_v = zip(*[(x, y, z, v) for x, y, z, v in raw_tensor_values if v != 0])
    raw_coords = np.array([raw_x, raw_y, raw_z], dtype=np.float64).T
    raw_coords = (raw_coords - raw_coords.mean(axis=0)) / (raw_coords.max(axis=0) - raw_coords.min(axis=0)).max()
    raw_x, raw_y, raw_z = raw_coords.T

    ax3.scatter(raw_x, raw_y, raw_z, c=raw_v, cmap='coolwarm', alpha=0.5)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.grid(False)

    raw_tensor_values = tensor_parser(tensor_string, args.density)
    raw_x, raw_y, raw_z, raw_v = zip(*[(x, y, z, v) for x, y, z, v in raw_tensor_values if v != 0])

    ax5 = fig.add_subplot(164, projection='3d')
    sc = ax5.scatter(raw_x, raw_y, raw_z, c=raw_v, cmap='coolwarm')
    plt.colorbar(sc, ax=ax5, orientation='horizontal')
    ax5.set_xlim(0, args.density)
    ax5.set_ylim(0, args.density)
    ax5.set_zlim(0, args.density)
    ax5.set_title('Raw Tensor (Non-Zero)')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')
    ax5.grid(False)

    ax6_values = tensor_parser(tensor_string, args.density)
    ax6_x, ax6_y, ax6_z, ax6_v = zip(*ax6_values)
    ax6 = fig.add_subplot(165, projection='3d')
    sc = ax6.scatter(ax6_x, ax6_y, ax6_z, c=ax6_v, cmap='coolwarm')
    plt.colorbar(sc, ax=ax6, orientation='horizontal')
    ax6.set_title('Tensor Values')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_zlabel('Z')
    ax6.grid(False)

    plt.show()

if __name__ == "__main__":
    main()
