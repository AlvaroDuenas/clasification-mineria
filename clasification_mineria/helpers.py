import os


def create_folder(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)