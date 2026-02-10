# config.py
import torch
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml

imgs_path = Path(r"C:\Users\images")
anno_path = Path(r"C:\Users\annotations")
yolo_base = Path(r"C:\Users\yolo_dataset")


def get_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def build_class2id():
    class2Id = {}
    current_id = 0

    for file in anno_path.glob("*.xml"):
        tree = ET.parse(file)
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in class2Id:
                class2Id[name] = current_id
                current_id += 1

    return class2Id


def give_datayaml():
    metadata = {
        "path": str(yolo_base),
        "train": "images/train",
        "val": "images/val",
        "names": ["with_mask", "without_mask", "mask_weared_incorrect"],
        "nc": 3
    }

    data_yaml = yolo_base / "data.yaml"
    with open(data_yaml, "w") as f:
        yaml.dump(metadata, f)

    return data_yaml
