import json
import os

def save_json_locally(filename_path, entities=[], unclean_description="", description="", prompt=""):
    json_data = {"command_name": prompt, "image_description:": description, "unclean_description": unclean_description, "bounding_boxs": []  }
    for name, rng, coordinates in entities:
        json_data["bounding_boxs"].append({
            "name": name,
            "range": rng,
            "coordinates": coordinates
        })

    # Write to JSON file
    with open(filename_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

def get_base_filename_without_extension(file_path):
    base_name, _ = os.path.splitext(os.path.basename(file_path))
    return base_name