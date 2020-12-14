import os


def create_folder(dir_path: str) -> None:
    """
    Creates a folder if doesnt exist
    Args:
        dir_path: the folder path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
