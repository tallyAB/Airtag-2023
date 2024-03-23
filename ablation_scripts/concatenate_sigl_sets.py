import os
from collections import OrderedDict

# Function to concatenate files with empty lines between different software
def concatenate_files_with_empty_lines(directory, output_file):
    software_data = OrderedDict()

    # Iterate over sorted files in the directory
    sorted_files = sorted(os.listdir(directory))
    for i, filename in enumerate(sorted_files):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            # Extract software name from filename
            software_name = filename.split('-')[0]

            # Read file content
            with open(file_path, 'r') as infile:
                content = infile.read().strip()

            # Append content to the dictionary with software name as key
            if software_name not in software_data:
                software_data[software_name] = []
            software_data[software_name].append(content)

            # Write an empty line if it's the last file of the software version
            if i == len(sorted_files) - 1 or not sorted_files[i + 1].startswith(software_name + '-'):
                software_data[software_name].append('')

    # Write concatenated content to the output file
    with open(output_file, 'w') as outfile:
        for files in software_data.values():
            for content in files:
                outfile.write(content + '\n')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-dir', action='store')
parser.add_argument('-out', action='store')
args = parser.parse_args()

# Directory containing files to concatenate
directory = args.dir

# Output file
output_file = args.out

# Concatenate files
concatenate_files_with_empty_lines(directory, output_file)

print("Files concatenated successfully.")
