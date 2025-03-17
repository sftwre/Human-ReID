import urllib.request
import os
import zipfile
import os
import shutil
from pathlib import Path


# Download the dataset
url = "https://www.kaggle.com/api/v1/datasets/download/pengcw1/market-1501"
filename = "market-1501.zip"
urllib.request.urlretrieve(url, filename)

# Unzip the file
with zipfile.ZipFile(filename, "r") as zip_ref:
    zip_ref.extractall("./data")

# Remove zip file
os.remove(filename)

src_root = Path("./data/Market-1501-v15.09.15")
dst_root = Path("./data/market-1501-grouped")


def group_by_person_id(src_dir_name: str, dst_dir_name: str):
    """
    Copies images from a sub-directory of the original dataset directory to a new
    sub-directory where images are grouped by person_id. This is to facilitate
    processing by the torchvision ImageFolder class.
    dst_dir_name will be created if it doesn't exists.

    Args:
        src_dir_name: name of source directory containing images to process
        dst_dir_name: name of destination directory where images are grouped by person id
    """

    src_dir = src_root / src_dir_name
    dst_dir = dst_root / dst_dir_name

    if not dst_dir.exists():
        os.makedirs(dst_dir)

    for filename in src_dir.iterdir():

        # skip non-image files
        if filename.suffix != ".jpg":
            continue

        """
        File naming schema for Market-1501 dataset
        
        <PersonID>_<CameraID><Sequence Number>_<Unique Frame Number>_<Detection BoxID>.jpg
        """

        person_id = filename.name.split("_")[0]
        src_path = filename
        dst_path = dst_dir / person_id

        if not dst_path.exists():
            os.mkdir(dst_path)

            # first image in training group is used for validation
            if dst_dir_name == "train":
                dst_path = dst_root / "val" / person_id
                os.makedirs(dst_path)

        shutil.copyfile(src_path, dst_path / filename.name)


# maps sub-directories in src_root to mirror directories in dst_root
data_dir_map = {
    "query": "query",
    "bounding_box_test": "gallery",
    "bounding_box_train": "train",
}

# src datt dir -> dst data dir
for sdd, ddd in data_dir_map.items():
    print(f"Grouping images in {src_root/sdd} -> {dst_root/ddd}")
    group_by_person_id(sdd, ddd)

shutil.rmtree(src_root)
print(f"Deleted {src_root}")
