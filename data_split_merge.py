import pandas as pd
import zipfile
import os

def split_file(filename, num_splits):
    # Read the original CSV file
    df = pd.read_csv(filename)

    # Determine the number of rows in each split file
    num_rows = len(df)
    split_size = num_rows // num_splits

    # Split the DataFrame into smaller DataFrames
    for i in range(num_splits):
        start_index = i * split_size
        end_index = (i + 1) * split_size if i != num_splits - 1 else num_rows
        split_df = df.iloc[start_index:end_index]
        
        # Save split DataFrame to CSV
        split_csv_filename = f'tensor_dataset_split_{i+1}.csv'
        split_df.to_csv(split_csv_filename, index=False)
        
        # Zip the CSV file
        zip_filename = f'tensor_dataset_split_{i+1}.zip'
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(split_csv_filename, os.path.basename(split_csv_filename))
        
        # Remove the original CSV file after zipping
        os.remove(split_csv_filename)

def merge_files(split_filenames, output_filename):
    # Read the split CSV files
    dataframes = []
    for zip_filename in split_filenames:
        with zipfile.ZipFile(zip_filename, 'r') as zipf:
            csv_filename = zipf.namelist()[0]
            zipf.extract(csv_filename)
            df = pd.read_csv(csv_filename)
            dataframes.append(df)
            # Remove the extracted CSV file after reading
            os.remove(csv_filename)

    # Concatenate the DataFrames
    df = pd.concat(dataframes)

    # Save the merged DataFrame to a new CSV file
    df.to_csv(output_filename, index=False)

# Usage
filename = 'tensor_electrostatic_dataset.csv'
num_splits = 6
split_file(filename, num_splits)

# To merge the files back together
# split_filenames = [f'tensor_dataset_split_{i+1}.zip' for i in range(num_splits)]
# output_filename = 'tensor_dataset_merged.csv'
# merge_files(split_filenames, output_filename)
