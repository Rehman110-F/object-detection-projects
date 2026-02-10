import xml.etree.ElementTree as ET

from config import build_class2id

class2Id = build_class2id()


def parse_annotations(path_file):
    tree = ET.parse(path_file)
    root = tree.getroot()

    w = int(root.find("size/width").text)
    h = int(root.find("size/height").text)

    yolo_lines = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text
        class_id = class2Id[class_name]

        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # YOLO normalization
        x_center = ((xmin + xmax) / 2) / w
        y_center = ((ymin + ymax) / 2) / h
        box_width = (xmax - xmin) / w
        box_height = (ymax - ymin) / h

        yolo_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    return yolo_lines

def labels_writer(yolo_lines, label_path):
    label_path.parent.mkdir(parents=True, exist_ok=True)

    with open(label_path, "w") as f:
        for line in yolo_lines:
            f.write(line + "\n")
    
