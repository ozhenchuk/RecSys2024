import os


def find_file(filename, max_iterations=5):
    current_dir = os.getcwd()
    for _ in range(max_iterations):
        potential_path = os.path.join(current_dir, filename)
        if os.path.isfile(potential_path):
            return potential_path
        # Move up one directory
        current_dir = os.path.dirname(current_dir)
    return None
