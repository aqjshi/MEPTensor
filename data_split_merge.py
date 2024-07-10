#takes a csv, split into 4 smaller files, create 4 files
import pandas as pd
import csv

def split_file(filename):
    # Read the original CSV file
    df = pd.read_csv(filename)

    # Determine the number of rows in each split file
    num_rows = len(df)
    split_size = num_rows // 4

    # Split the DataFrame into four smaller DataFrames
    df1 = df.iloc[:split_size]
    df2 = df.iloc[split_size:2*split_size]
    df3 = df.iloc[2*split_size:3*split_size]
    df4 = df.iloc[3*split_size:]

    # Save each smaller DataFrame to a new CSV file
    df1.to_csv('tensor_dataset_split_1.csv', index=False)
    df2.to_csv('tensor_dataset_split_2.csv', index=False)
    df3.to_csv('tensor_dataset_split_3.csv', index=False)
    df4.to_csv('tensor_dataset_split_4.csv', index=False)


def merge_files(file1, file2, file3, file4):
    # Read the four split CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    df4 = pd.read_csv(file4)

    # Concatenate the DataFrames
    df = pd.concat([df1, df2, df3, df4])

    # Save the merged DataFrame to a new CSV file
    df.to_csv('tensor_dataset_merged.csv', index=False)


# Usage
split_file('tensor_dataset_v3.csv')
