from pathlib import Path
from tqdm import tqdm
import random
import shutil

from utils import parse_annotations , labels_writer
from config import imgs_path , img_list , anno_path,annotations_list
# print(f"The total images are: {len(img_list)} , The total annotations are : {len(annotations_list)}")
# print(img_list[:5])
yolo_base = Path(r"C:\Users\yolo_dataset")
yolo_base.mkdir(parents=True ,exist_ok=True)
(yolo_base / "images" / "train").mkdir(parents=True ,exist_ok=True)
(yolo_base / "images" / "val").mkdir(parents=True ,exist_ok=True)
(yolo_base / "labels" / "train").mkdir(parents=True ,exist_ok=True)
(yolo_base / "labels" / "val").mkdir(parents=True ,exist_ok=True)

# now the time is to take all the data and then paste into this above hierarchy
# create a class mapping to ids so 

# dataset creation
random.seed(42)
train_frac= 0.8

for img in tqdm(img_list):

    # decide the split 
    split = "train" if random.random() < train_frac else "val"

    # corresponding xml
    annotation = anno_path / f"{img.stem}.xml"

    try:
        yolo_lines = parse_annotations(annotation)
    except Exception as e:
        print(f'Failed to parse "{img.stem}". Skipping.')
        print(e)
        continue

    # Write label
    label_dest = yolo_base / "labels" / split / f"{img.stem}.txt"
    labels_writer(yolo_lines , label_dest)

    # Copy image
    image_dest = yolo_base/ "images" / split / img.name
    image_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(img, image_dest)