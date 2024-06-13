from dotenv import dotenv_values
from file import find_file


def config():
    return dotenv_values(find_file(".env"))
