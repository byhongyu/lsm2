from datasets import load_dataset


if __name__ == "__main__":
    ds = load_dataset("UCSC-VLAA/Recap-DataComp-1B", split="preview")

    print(type(ds))