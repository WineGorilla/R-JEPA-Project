import tensorflow_datasets as tfds

DATA_DIR = "/data/mi2-interns/ruiyu/R-JEPA/data/tfds"

print("Loading MOVi-C from local disk...")

builder = tfds.builder(
    "movi_c/256x256",
    data_dir=DATA_DIR,
)

print("Dataset:", builder.info.full_name)
print("Splits:", builder.info.splits)

ds = builder.as_dataset(
    split="train[:1]",
    shuffle_files=False,
)

for sample in ds.take(1):

    print("\nSample loaded successfully.")

    print("Keys:")
    for key in sample.keys():
        print(" ", key)

    print("\nVideo shape:")
    print(sample["video"].shape)

    if "segmentations" in sample:
        print("\nSegmentation shape:")
        print(sample["segmentations"].shape)

print("\nLOCAL MOVi-C TEST PASSED.")