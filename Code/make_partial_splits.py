from random import shuffle, seed

if __name__ == "__main__":
    seed(0)
    splits_path = "/mnt/data/enmap/dataset/splits/easy/"
    new_split_path = "/mnt/data/enmap/dataset/splits/easy_256/"
    divisor = 256
    splits = ["train", "val", "test"]
    for split in splits:
        csv_path = splits_path + split + ".csv"
        lines = []
        with open(csv_path) as file:
            lines = [line.rstrip() for line in file]
            print(f"Input for {split} has {len(lines)} lines.")
            shuffle(lines)
            lines = lines[:len(lines)//divisor]
        new_csv_path = new_split_path + split + ".csv"
        with open(new_csv_path, mode="wt", encoding="utf-8") as file:
            file.write("\n".join(lines))
            print(f"Output for {split} has {len(lines)} lines.")
