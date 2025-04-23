"""
os_tools/oversampler.py

Module for handling class imbalance through oversampling and data augmentation.
Supports CLI generation of augmentation plans (maps), programmatic oversampling, and class analysis tools.
"""

# Built-in and STL
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

# 3rd party
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ( #type: ignore
    ImageDataGenerator,
    img_to_array,
    array_to_img,
    load_img,
)

# Root package
from deep.constants import IMAGE_DIR, RESOURCES_DIR, DATA_DIR, METADATA_FILE, BINARY_UPSAMPLE, FAMILY_UPSAMPLE

# Transformations dict for ImageGenerator
TRANSFORM_GENERATORS: Dict[str, ImageDataGenerator] = {
    "flip_lr": ImageDataGenerator(horizontal_flip=True),
    "bright_plus": ImageDataGenerator(preprocessing_function=lambda x: x * 1.10),
    "bright_minus": ImageDataGenerator(preprocessing_function=lambda x: x * 0.90),
    "sat_plus": ImageDataGenerator(preprocessing_function=lambda x: tf.image.adjust_saturation(x, 1.15)),
    "sat_minus": ImageDataGenerator(preprocessing_function=lambda x: tf.image.adjust_saturation(x, 0.85)),
    "zoom_in": ImageDataGenerator(zoom_range=[1.0, 1.2]),
    "zoom_out": ImageDataGenerator(zoom_range=[0.8, 1.0]),
    "shift": ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1),
    "rotate_15": ImageDataGenerator(rotation_range=15),
    "rotate_30": ImageDataGenerator(rotation_range=30),
    "rotate_45": ImageDataGenerator(rotation_range=45),
    "rotate_60": ImageDataGenerator(rotation_range=60),
    "rotate_90": ImageDataGenerator(rotation_range=90),
}

def compute_effective_class_weights(
    df: pd.DataFrame,
    label_column: str,
    beta: float = 0.999,
    normalize: bool = True
) -> Dict[str, float]:
    
    def effective_num(n: int, beta: float) -> float:
        return (1 - beta**n) / (1 - beta)

    label_counts = df[label_column].value_counts()
    effective_nums = label_counts.apply(lambda n: effective_num(n, beta))
    total_effective = effective_nums.sum()
    class_weights = (total_effective / effective_nums).to_dict()

    if normalize:
        mean_weight = np.mean(list(class_weights.values()))
        class_weights = {cls: w / mean_weight for cls, w in class_weights.items()}

    return class_weights

def compute_imbalance_ratio(
    df: pd.DataFrame,
    label_column: str
) -> Dict[str, float]:
    counts = df[label_column].value_counts()
    max_count = counts.max()
    return {label: max_count / count for label, count in counts.items()}

def map_oversample(
    df: pd.DataFrame,
    label: str,
    target_ratio: float,
    min_samples: int
) -> List[Dict[str, str]]:

    counts = df[label].value_counts()
    max_count = counts.max()
    majority_label = counts.idxmax()
    plan = []

    for cls_label, count in counts.items():
        if cls_label == majority_label:
            continue

        current_ratio = max_count / count
        if (current_ratio <= target_ratio) and (count > min_samples):
            continue

        target_count = max(int(max_count / target_ratio), (min_samples - count))
        needed = target_count - count
        if needed <= 0:
            continue

        class_df = df[df[label] == cls_label]
        transform_keys = list(TRANSFORM_GENERATORS.keys())
        transform_idx = 0

        while needed > 0 and transform_idx < len(transform_keys):

            transform_key = transform_keys[transform_idx]
            n_samples = min(needed, len(class_df))

            selected = class_df.sample(n_samples)
            selected['transform_key'] = transform_key

            if len(plan) == 0:
                plan = selected
            else:
                plan = pd.concat([plan, selected], ignore_index=True)

            needed -= n_samples
            transform_idx += 1

    return plan

def _apply_transformation(
        img: Any
    , transformation_key: str
    ) -> Any:
    if transformation_key not in TRANSFORM_GENERATORS:
        raise ValueError(f"Unknown transformation key: {transformation_key}")

    datagen = TRANSFORM_GENERATORS[transformation_key]
    img_array = img_to_array(img)
    img_array = img_array.reshape((1,) + img_array.shape)
    aug_iter = datagen.flow(img_array, batch_size=1)
    aug_img_array = next(aug_iter)[0].astype(np.uint8)
    return array_to_img(aug_img_array)

def oversample_labels():
    instructions = [
        ['is_animal', 'binary_oversampled_data', BINARY_UPSAMPLE],
        ['family', 'family_oversampled_data', FAMILY_UPSAMPLE]
    ]
    for label, output_name, plan in instructions:

        output_name = Path(output_name).stem

        with open(plan, "r") as f:
            plan_data = json.load(f)

        augmented_rows = []

        for entry in plan_data:
            source = IMAGE_DIR / entry["file_path"]
            filename = f"{source.stem}_{entry[label]}_{entry['transform_key']}{source.suffix}"
            destination = IMAGE_DIR / filename

            try:
                img = load_img(source)
                new_img = _apply_transformation(img, entry["transform_key"])
                new_img.save(destination)

                augmented_rows.append({
                    "rare_species_id": entry['rare_species_id'],
                    "file_path": str(filename)
                })

            except Exception as e:
                print(f"Failed on {source.name}: {e}")

        df_aug = pd.DataFrame(augmented_rows)
        filepath = DATA_DIR / f"{output_name}.csv"
        df_aug.to_csv(filepath, mode='a', index=False)

# CLI for generating augmentation plan only
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate oversampling augmentation plan only.")
    parser.add_argument('--output_name', type=str, required=True, help="Base name for output files")
    parser.add_argument('--label', type=str, required=True, help="Label column for class balancing")
    parser.add_argument('--min_sample', type=int, default=30, help="Minimum number of samples per class")
    parser.add_argument('--target_ratio', type=float, default=3.0, help="Desired maximum imbalance ratio")
    
    args = parser.parse_args()

    if args.label == "is_animal":
        df_meta = pd.read_csv(METADATA_FILE)
        df_bin = pd.read_csv(DATA_DIR / "binary_oversample_data.csv")
        df = pd.concat([df_meta, df_bin], ignore_index=True)
    else:
        df = pd.read_csv(METADATA_FILE)

    df = df[['rare_species_id', 'file_path', f'{args.label}']]
    plan = pd.DataFrame(map_oversample(df, args.label, args.target_ratio, args.min_sample))
    plan_path = RESOURCES_DIR / f"{Path(args.output_name).stem}.json"
    plan_path.write_text(plan.to_json(orient='records', indent=2))
    print(f"Saved oversample map with {len(plan)} entries to {plan_path}")

    # Actual calls made during this project.
    # python album_augmenter.py --output_name binary_upsample_map --label is_animal --min_sample 150 --target_ratio 1
    # python album_augmenter.py --output_name family_upsample_map --label family --min_sample 150 --target_ratio 2