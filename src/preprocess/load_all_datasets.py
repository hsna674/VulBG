import pickle
import os


def load_src_files(src_path, is_vul=0):
    final_data = []
    if not os.path.exists(src_path):
        print(f"Warning: Path {src_path} does not exist")
        return final_data

    for file in os.listdir(src_path):
        if not file.endswith(".c"):
            continue
        filepath = os.path.join(src_path, file)
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                item = {
                    "file": file,
                    "code": content,
                    "vul": is_vul,
                    "label": is_vul,
                    "slices": [],
                    "dataset": os.path.basename(os.path.dirname(src_path)),
                }
                final_data.append(item)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
    return final_data


def create_dummy_slices(final_data):
    for item in final_data:
        code_lines = item["code"].split("\n")
        slices = []
        for i in range(0, len(code_lines), 15):
            slice_lines = code_lines[i : i + 20]
            if len(slice_lines) > 5:
                slices.append("\n".join(slice_lines))

        # Limit to 20 slices as per the original code
        item["slices"] = slices[:20] if slices else [item["code"][:1000]]

    return final_data


if __name__ == "__main__":
    base_path = "VulBG/dataset/raw"

    datasets = ["ffmpeg", "qemu", "devign"]
    all_data = []

    for dataset in datasets:
        vul_path = f"{base_path}/{dataset}/Vul"
        novul_path = f"{base_path}/{dataset}/NoVul"

        print(f"Loading {dataset} dataset...")

        vul_data = load_src_files(vul_path, 1)
        print(f"Loaded {len(vul_data)} vulnerable files from {dataset}")

        novul_data = load_src_files(novul_path, 0)
        print(f"Loaded {len(novul_data)} non-vulnerable files from {dataset}")

        all_data.extend(vul_data)
        all_data.extend(novul_data)

    print(f"Total files loaded: {len(all_data)}")

    print("Creating code slices...")
    all_data = create_dummy_slices(all_data)

    final_data = [item for item in all_data if item["slices"]]
    print(f"Final dataset size: {len(final_data)}")

    output_path = "VulBG/dataset/processed/combined_dataset.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(final_data, f)

    print(f"Dataset saved to {output_path}")

    vul_count = sum(1 for item in final_data if item["vul"] == 1)
    novul_count = len(final_data) - vul_count
    print(f"Vulnerable: {vul_count}, Non-vulnerable: {novul_count}")
