# Allows option to remove DNS, Browser or both --dns --browser
# AblationOne: Remove both DNS and Browser logs
# Ablation Two: Remove Browser logs only

# Take original training_preprocessed, training, and benign files and remove according to args
# Take original test file and remove according to args, update GT file accordingly
# Uses new pre_processed file to create new vocab (Done by calling bash script)

import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dns', action='store_true', default=False, dest='rem_dns')
parser.add_argument('--browser', action='store_true', default=False, dest='rem_browser')
args = parser.parse_args()
if args.rem_dns and args.rem_browser:
     ablOne = True
else:
    if not args.rem_browser:
        print("Error: Remove either DNS, or Borwser or both")
        exit()
    ablOne = False
    
if ablOne:
    suffix = 'AblationOne/'
else:
    suffix = 'AblationTwo/'

starting_path = '/root/AirTag/training_data/'
pp_files = ['training_preprocessed_logs_S1-CVE-2015-5122_windows', 'training_preprocessed_logs_S2-CVE-2015-3105_windows', 'training_preprocessed_logs_S3-CVE-2017-11882_windows', 'training_preprocessed_logs_S4-CVE-2017-0199_windows_py']
train_files = ['S1_train', 'S2_train', 'S3_train', 'S4_train']
benign_files = ['S1_benign', 'S2_benign', 'S3_benign', 'S4_benign']
test_files = ['S1_test', 'S2_test', 'S3_test', 'S4_test']

for item in pp_files+train_files+benign_files:
    org_file_path = starting_path+item
    filtered_file_path = starting_path + suffix + item
    
    with open(org_file_path, 'r') as original_file, open(filtered_file_path, 'w') as filtered_file:
        lines = original_file.readlines()
        for line in lines:
            if ablOne:
                if line.strip().endswith('LA-') or line.strip().endswith('LA+'):
                    filtered_file.write(line)
            else: #AblationTwo
                if line.strip().endswith('LA-') or line.strip().endswith('LA+') or line.strip().endswith('LD-') or line.strip().endswith('LD+'):
                    filtered_file.write(line)
                
# Test files

for j,test_file in enumerate(test_files):
    indices_removed = []
    
    org_file_path = starting_path+test_file
    filtered_file_path = starting_path + suffix + test_file
    with open(org_file_path, 'r') as original_file, open(filtered_file_path, 'w') as filtered_file:
        lines = original_file.readlines()
        for i, line in enumerate(lines):
            if ablOne:
                if line.strip().endswith('LA-') or line.strip().endswith('LA+'):
                    filtered_file.write(line)
                else:
                    indices_removed.append(i)
            else: #AblationTwo
                if line.strip().endswith('LA-') or line.strip().endswith('LA+') or line.strip().endswith('LD-') or line.strip().endswith('LD+'):
                    filtered_file.write(line)
                else:
                    indices_removed.append(i)

    # Load gt data
    gt_indices = np.load('/root/Airtag/ground_truth/S'+str(j+1)+'_number_.npy')
    gt_indices = list(set(gt_indices))
    gt_indices = [int(x) for x in gt_indices]
    gt_indices.sort()
    
    for deleted_index in indices_removed:    
        # Remove the deleted index from the indices list
        if deleted_index in gt_indices:
            gt_indices.remove(deleted_index)
    
    red_dict = {item: 0 for item in gt_indices} # This dict keeps count of the amount the index in final gt needs to be reduced by
    for deleted_index in indices_removed:
        # Rebuild the entire indices list
        for index in gt_indices:
            if index > deleted_index:
                red_dict[index] = red_dict[index] + 1
                
    # Update indices
    updated_indices = []
    for index in gt_indices:
        updated_indices.append(index-red_dict[index])    

    # Write updated indices to the correct gt folder
    np.save('/root/Airtag/ground_truth/' + suffix +'S'+str(j+1)+'_number_.npy',updated_indices)
