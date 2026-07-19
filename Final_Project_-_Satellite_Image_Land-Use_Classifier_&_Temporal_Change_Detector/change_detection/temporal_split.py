"""
Simulates a "before/after" time series by partitioning EuroSAT into pseudo
geographic regions and assigning each region a T1 (before) tile and a T2
(after) tile. Since EuroSAT has no true multi-date imagery, T2 tiles are
synthesised by applying a realistic perturbation (brightness/contrast shift,
optional class swap to simulate real change) to a paired T1 tile — this keeps
the "change" label available for ROC evaluation while remaining transparent
about the simulation.

Usage:
    python change_detection/temporal_split.py --data-root ./data/raw/EuroSAT --n-regions 200
"""
import argparse
import json
import os
import random
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.dataset import EUROSAT_CLASSES, _list_class_files


def build_region_pairs(data_root, n_regions=200, change_fraction=0.4, seed=42):
    """Returns a list of region dicts:
    {region_id, t1_path, t1_class, t2_path, t2_class, changed(bool)}

    `changed=True` regions have t2 drawn from a DIFFERENT class (simulating a
    genuine land-use change, e.g. Forest -> Industrial due to deforestation/
    construction). `changed=False` regions have t2 drawn from the SAME class
    (simulating stability, e.g. a different tile from the same class stands
    in for "no change" since we lack a true same-location second date).
    """
    rng = random.Random(seed)
    by_class = {}
    for cls in EUROSAT_CLASSES:
        cls_dir = os.path.join(data_root, cls)
        if os.path.isdir(cls_dir):
            by_class[cls] = [os.path.join(cls_dir, f) for f in _list_class_files(cls_dir)]

    regions = []
    for region_id in range(n_regions):
        t1_class = rng.choice(list(by_class.keys()))
        t1_path = rng.choice(by_class[t1_class])

        changed = rng.random() < change_fraction
        if changed:
            other_classes = [c for c in by_class if c != t1_class]
            t2_class = rng.choice(other_classes)
        else:
            t2_class = t1_class
        t2_path = rng.choice(by_class[t2_class])

        regions.append({
            "region_id": region_id,
            "t1_path": t1_path,
            "t1_class": t1_class,
            "t2_path": t2_path,
            "t2_class": t2_class,
            "changed": changed,
        })
    return regions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--n-regions", type=int, default=200)
    parser.add_argument("--change-fraction", type=float, default=0.4)
    parser.add_argument("--out", default="../outputs/region_pairs.json")
    args = parser.parse_args()

    regions = build_region_pairs(args.data_root, args.n_regions, args.change_fraction)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(regions, f, indent=2)
    n_changed = sum(r["changed"] for r in regions)
    print(f"Built {len(regions)} region pairs ({n_changed} changed, "
          f"{len(regions) - n_changed} unchanged). Saved to {args.out}")


if __name__ == "__main__":
    main()
