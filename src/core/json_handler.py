import json
import os
from .logger import logger

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
data_dir = os.path.abspath(data_dir)

os.makedirs(data_dir, exist_ok=True)

json_file_path = os.path.join(data_dir, "darts_data.json")
 

def setup_json():
    if not os.path.exists(json_file_path):
        with open(json_file_path, mode="w") as file:
            json.dump([], file, indent=4)
        logger.info("Created new JSON file: %s", json_file_path)


def log_to_json(data):
    try:
        with open(json_file_path, mode='r+') as file:
            try:
                file_data = json.load(file)  
            except json.JSONDecodeError:
                logger.warning("JSON file is corrupted or empty. Resetting the file.")
                file_data = []  
            
            file_data.append(data)  
            file.seek(0)           
            json.dump(file_data, file, indent=4)  
    except FileNotFoundError:
        logger.warning("JSON file not found. Creating a new file.")
        setup_json()  
        log_to_json(data)