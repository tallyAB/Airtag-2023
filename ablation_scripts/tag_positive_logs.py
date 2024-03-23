import numpy as np

def tag_logs_with_ground_truth(file_path, pattern):
    
    with open(file_path, 'r+') as f:
        preprocessed_logs = f.readlines()
        f.seek(0)  # Rewind the file pointer

        for log_entry in preprocessed_logs:
            if pattern in log_entry: # Add + sign if found
                f.write(log_entry.strip()[:-1] + '+\n')
            else:
                f.write(log_entry)
        f.truncate()  # Truncate the file in case it's shorter now

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', action='store')
    parser.add_argument('-pattern', action='store')
    args = parser.parse_args()
    
    file_path = args.file
    pattern = args.pattern
    print(pattern)
    tag_logs_with_ground_truth(file_path, pattern)
