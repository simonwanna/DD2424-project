import random
from pathlib import Path
from PIL import Image
from torchvision import transforms
import argparse


INPUT_SIZE = 224
RESIZE_SIZE = 256  # shorter side -> 256, then center-crop to 224x224
RAW_DATA_DIR = Path("./data")
IMAGES_DIR = RAW_DATA_DIR / 'images'
METADATA_FILE = RAW_DATA_DIR / 'annotations/list.txt'
OUTPUT_DIR = Path("./data")
SPLIT_RATIOS = {'train': 0.7, 'val': 0.2, 'test': 0.1}

preprocess = transforms.Compose([
    # transforms.Resize(RESIZE_SIZE),
    transforms.CenterCrop(INPUT_SIZE),
])


def get_labels(cls_type: str = 'species'):
    """
    Get labels for the dataset.
    :param cls_type: 'species' or 'breed'
    :return: dict of label maps
    """
    if cls_type == 'species':
        return {'1': 'cat', '2': 'dog'}
    elif cls_type == 'breed':
        return  {'1': 'Abyssinian', '2': 'american_bulldog', '3': 'american_pit_bull_terrier', '4': 'basset_hound', 
                 '5': 'beagle', '6': 'Bengal', '7': 'Birman', '8': 'Bombay', '9': 'boxer', '10': 'British_Shorthair', 
                 '11': 'chihuahua', '12': 'Egyptian_Mau', '13': 'english_cocker_spaniel', '14': 'english_setter', 
                 '15': 'german_shorthaired', '16': 'great_pyrenees', '17': 'havanese', '18': 'japanese_chin', 
                 '19': 'keeshond', '20': 'leonberger', '21': 'Maine_Coon', '22': 'miniature_pinscher', '23': 'newfoundland', 
                 '24': 'Persian', '25': 'pomeranian', '26': 'pug', '27': 'Ragdoll', '28': 'Russian_Blue', 
                 '29': 'saint_bernard', '30': 'samoyed', '31': 'scottish_terrier', '32': 'shiba_inu', '33': 'Siamese', 
                 '34': 'Sphynx', '35': 'staffordshire_bull_terrier', '36': 'wheaten_terrier', '37': 'yorkshire_terrier'}
    else:
        raise ValueError(
            "Invalid classification type. Use 'species' or 'breed'.")


def load_metadata(metadata_file: Path, mapping: dict, cls_type: str = 'species'):
    """
    Parse list.txt to map filenames to species labels.
    Each line has 4 parts: <Image> <CLASS-ID> <SPECIES> <BREED-ID>
    Returns a dict: {label: [filenames]}.
    """
    label_groups = {label: [] for label in mapping.values()}
    with open(metadata_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            img_name, class_id, species_id, breed_id = parts
            fname = img_name + '.jpg'
            
            if cls_type == 'species':
                label_id = species_id
            elif cls_type == 'breed':
                label_id = class_id
            
            label = mapping.get(label_id)
            if label:
                label_groups[label].append(fname)
    return label_groups


def process_and_save(src_path: Path, dst_path: Path):
    """Transform image to correct size and save it."""
    img = Image.open(src_path).convert('RGB')
    img_processed = preprocess(img)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img_processed.save(dst_path)


def split_and_preprocess(label: str, filenames: list):
    """Split list of filenames into train/val/test, process and save"""
    random.shuffle(filenames)
    n = len(filenames)

    n_train = int(n * SPLIT_RATIOS['train'])
    n_val   = int(n * SPLIT_RATIOS['val'])

    splits = {
        'train': filenames[:n_train],
        'val':   filenames[n_train:n_train + n_val],
        'test':  filenames[n_train + n_val:],
    }

    for split, fnames in splits.items():
        for fname in fnames:
            src = IMAGES_DIR / fname
            dst = OUTPUT_DIR / split / label / fname
            process_and_save(src, dst)

    counts = {s: len(fnames) for s, fnames in splits.items()}
    print("\033[94m" + f"Label '{label}': {counts['train']} train, {counts['val']} val, {counts['test']} test images." + "\033[0m")


if __name__ == '__main__':
    random.seed(42)
    parser = argparse.ArgumentParser(description='Preprocess images for training.')
    cls_type = parser.add_argument('--cls_type', type=str, default='species', choices=['species', 'breed'],
                                   help='Type of classification (species or breed)')
    args = parser.parse_args()
    cls_type = args.cls_type; print(f"Preprocessing for {cls_type} classification.")
    
    groups = load_metadata(METADATA_FILE, get_labels(cls_type), cls_type)
    print(f"Found {len(groups)} labels in metadata file.")

    for label, files in groups.items():
        split_and_preprocess(label, files)

    print("Preprocessing complete. Data saved in:")
    for split in SPLIT_RATIOS:
        print(f"  {split}: {OUTPUT_DIR / split}")
