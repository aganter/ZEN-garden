import os
import json

directory_path = ('/Users/Ludwig/Documents/ETH/Master/Master_Thesis/ZEN-garden/data/data_II_copy/set_technologies')

files = os.listdir('/data/Ammonia_Pathways/set_technologies')

print(f"{directory_path}")

for root, dirs, files in os.walk(directory_path):

    for file_name in files:
        file_path = os.path.join(root, file_name)

        if file_name.endswith('.json'):
            with open(file_path, 'r') as file:
                data = json.load(file)

                new_data = {list(entry.keys())[0]: entry[list(entry.keys())[0]] for entry in data}

                with open(file_path, 'w') as file:
                    json.dump(new_data, file, indent=2)

                print(f"Converted {file_name}")