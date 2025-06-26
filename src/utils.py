import os

def path_builder(folder_name:str, path_name: str) -> str:
    """
    Build an absolute path for a given path name.
    
    Args:
        path_name (str): Name of the path to build
    
    Returns:
        str: Absolute path
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the project root
    project_root = os.path.dirname(script_dir)
    # Build the absolute path to the combined data file
    built_path_str = os.path.join(project_root, folder_name, path_name)

    return built_path_str