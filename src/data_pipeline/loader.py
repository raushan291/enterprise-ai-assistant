import os


def load_data(data_dir: str) -> list[str]:
    """Load all `.txt` files from a given directory into a list of strings."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    texts = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts
