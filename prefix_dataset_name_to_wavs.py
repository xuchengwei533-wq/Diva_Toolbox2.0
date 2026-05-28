from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"
SEPARATOR = "__"


def iter_dataset_wavs():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    for dataset_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        dataset_name = dataset_dir.name
        for wav_path in sorted(dataset_dir.rglob("*.wav")):
            if wav_path.is_file():
                yield dataset_name, wav_path


def main() -> None:
    renamed_count = 0

    for dataset_name, wav_path in iter_dataset_wavs():
        prefix = f"{dataset_name}{SEPARATOR}"
        if wav_path.name.startswith(prefix):
            print(f"[=] Keep: {wav_path.relative_to(DATA_DIR)}")
            continue

        new_path = wav_path.with_name(f"{prefix}{wav_path.name}")
        if new_path.exists():
            print(f"[!] Skip, target already exists: {new_path.relative_to(DATA_DIR)}")
            continue

        wav_path.rename(new_path)
        renamed_count += 1
        print(f"{wav_path.relative_to(DATA_DIR)} -> {new_path.relative_to(DATA_DIR)}")

    print(f"[+] Done. Renamed {renamed_count} wav files.")


if __name__ == "__main__":
    main()
