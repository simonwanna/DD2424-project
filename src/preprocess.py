import random
from pathlib import Path
from PIL import Image
from torchvision import transforms


INPUT_SIZE = 224
RESIZE_SIZE = 256  # shorter side -> 256, then center-crop to 224x224
RAW_DATA_DIR = Path("./data")
IMAGES_DIR = RAW_DATA_DIR / 'images'
METADATA_FILE = RAW_DATA_DIR / 'annotations/list.txt'
OUTPUT_DIR = Path("./data")
SPLIT_RATIOS = {'train': 0.7, 'val': 0.2, 'test': 0.1}

preprocess = transforms.Compose([
    transforms.Resize(RESIZE_SIZE),
    transforms.CenterCrop(INPUT_SIZE),
])

SPECIES_MAP = {'1': 'cat', '2': 'dog'}


def load_metadata(metadata_file: Path):
    """
    Parse list.txt to map filenames to species labels.
    Each line has 4 parts: <Image> <CLASS-ID> <SPECIES> <BREED-ID>
    Returns a dict: { 'cat': [filenames], 'dog': [filenames] }
    """
    species_groups = {label: [] for label in SPECIES_MAP.values()}
    with open(metadata_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            img_name, _, species_id, _ = parts
            fname = img_name + '.jpg'
            label = SPECIES_MAP.get(species_id)
            if label:
                species_groups[label].append(fname)
    return species_groups


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
    species_groups = load_metadata(METADATA_FILE)
    for label, files in species_groups.items():
        split_and_preprocess(label, files)

    print("Preprocessing complete. Data saved in:")
    for split in SPLIT_RATIOS:
        print(f"  {split}: {OUTPUT_DIR / split}")
