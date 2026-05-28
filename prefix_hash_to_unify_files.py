from pathlib import Path


TARGET_DIR = Path(__file__).resolve().parent / "data" / "Unify" / "同音高"


def main() -> None:
    if not TARGET_DIR.exists():
        raise FileNotFoundError(f"Directory not found: {TARGET_DIR}")

    renamed_count = 0
    for path in sorted(TARGET_DIR.iterdir()):
        if not path.is_file():
            continue

        if path.name.startswith("##"):
            print(f"[=] Keep: {path.name}")
            continue
        if path.name.startswith("#"):
            new_name = f"#{path.name}"
        else:
            new_name = f"##{path.name}"

        new_path = path.with_name(new_name)
        if new_path.exists():
            print(f"[!] Skip, target already exists: {new_path.name}")
            continue

        path.rename(new_path)
        renamed_count += 1
        print(f"{path.name} -> {new_path.name}")

    print(f"[+] Done. Renamed {renamed_count} files.")


if __name__ == "__main__":
    main()
