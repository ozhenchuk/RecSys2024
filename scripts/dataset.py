import os
import requests
from zipfile import ZipFile
from io import BytesIO
from load_env import config


def download_dataset():
    conf = config()
    url = str(conf["DATASET_DOWNLOAD_URL"])
    extract_to = "data"
    if os.path.exists(extract_to) and os.listdir(extract_to):
        print(f"The directory {extract_to} already contains files. Skipping download.")
        return

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    response = requests.get(url)
    if response.status_code == 200:
        with ZipFile(BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Downloaded and extracted ZIP file from {url} to {extract_to}")
    else:
        print(
            f"Failed to download ZIP file from {url}. Status code: {response.status_code}"
        )


if __name__ == "__main__":
    download_dataset()
