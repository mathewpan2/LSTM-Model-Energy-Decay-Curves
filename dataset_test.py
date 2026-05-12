import argparse
import json
import os
import re

import numpy as np
import pandas as pd


def check_local_dataset(dataset_root: str, limit: int) -> None:
    csv_path = os.path.join(dataset_root, "roomFeaturesDataset.csv")
    edc_dir = os.path.join(dataset_root, "EDC")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.isdir(edc_dir):
        raise FileNotFoundError(f"EDC folder not found: {edc_dir}")

    df = pd.read_csv(csv_path)
    id_col = "ID" if "ID" in df.columns else df.columns[0]

    print(f"Loaded CSV: {csv_path}")
    print(f"Rows: {len(df)} | Columns: {list(df.columns)}")

    missing = []
    for idx, row in df.head(limit).iterrows():
        sample_id = str(row[id_col])
        edc_path = os.path.join(edc_dir, f"{sample_id}.npy")
        if not os.path.exists(edc_path):
            missing.append(edc_path)
            continue

        edc = np.load(edc_path)
        print(f"[{idx}] {sample_id}: shape={edc.shape}, dtype={edc.dtype}, min={float(np.min(edc)):.6f}, max={float(np.max(edc)):.6f}")

    if missing:
        print("Missing EDC files:")
        for path in missing:
            print(f"  - {path}")
    else:
        print(f"All checked EDC files exist for the first {min(limit, len(df))} rows.")


def check_acoustic_rooms_dataset(dataset_root: str, limit: int) -> None:
    ir_root = os.path.join(dataset_root, "single_channel_ir")
    meta_root = os.path.join(dataset_root, "metadata")
    depth_root = os.path.join(dataset_root, "depth_map")

    if not os.path.isdir(ir_root):
        raise FileNotFoundError(f"IR root not found: {ir_root}")
    if not os.path.isdir(meta_root):
        raise FileNotFoundError(f"Metadata root not found: {meta_root}")

    wav_paths = []
    for room_type in sorted(os.listdir(ir_root)):
        room_type_path = os.path.join(ir_root, room_type)
        if not os.path.isdir(room_type_path):
            continue

        for room_id in sorted(os.listdir(room_type_path)):
            room_path = os.path.join(room_type_path, room_id)
            if not os.path.isdir(room_path):
                continue
            for file_name in sorted(os.listdir(room_path)):
                if file_name.endswith("_hybrid_IR.wav"):
                    wav_paths.append((room_type, room_id, os.path.join(room_path, file_name)))

    if not wav_paths:
        raise RuntimeError("No RIR wav files found.")

    print(f"Found {len(wav_paths)} wav files under {ir_root}")
    for i, (room_type, room_id, wav_path) in enumerate(wav_paths[:limit]):
        stem = os.path.basename(wav_path).replace("_hybrid_IR.wav", "")
        meta_path = os.path.join(meta_root, room_type, room_id, f"{stem}.json")
        if not os.path.exists(meta_path):
            print(f"[{i}] missing metadata: {meta_path}")
            continue

        with open(meta_path, "r") as meta_file:
            meta = json.load(meta_file)
        rir, sr = None, None
        try:
            import soundfile as sf

            rir, sr = sf.read(wav_path)
            if rir.ndim > 1:
                rir = rir[:, 0]
            rir = np.asarray(rir, dtype=np.float32)
        except Exception as exc:
            print(f"[{i}] failed to load wav {wav_path}: {exc}")
            continue

        depth_info = "no depth map"
        if os.path.isdir(depth_root):
            receiver_match = re.search(r"_R(\d+)$", stem)
            if receiver_match is not None:
                depth_path = os.path.join(depth_root, room_type, room_id, f"{int(receiver_match.group(1))}.npy")
                if os.path.exists(depth_path):
                    depth = np.load(depth_path)
                    depth_info = f"depth shape={depth.shape}"
                else:
                    depth_info = "missing depth map"

        print(
            f"[{i}] {stem}: wav shape={rir.shape}, sr={sr}, "
            f"src={meta.get('src_loc', 'n/a')}, rec={meta.get('rec_loc', 'n/a')}, {depth_info}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick dataset sanity check.")
    parser.add_argument(
        "--dataset-root",
        default="/mnt/code/code/AcousticRooms/",
        help="Path to the local dataset root or AcousticRooms root.",
    )
    parser.add_argument("--limit", type=int, default=5, help="Number of samples to inspect.")
    parser.add_argument(
        "--mode",
        choices=["local", "acoustic_rooms"],
        default="local",
        help="Which dataset layout to verify.",
    )
    args = parser.parse_args()

    if args.mode == "local":
        check_local_dataset(args.dataset_root, args.limit)
    else:
        check_acoustic_rooms_dataset(args.dataset_root, args.limit)


if __name__ == "__main__":
    main()