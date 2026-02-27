import numpy as np
import lmdb
import pickle
from tqdm import tqdm

PATCH_PATH = "./dataset/patches.npy"
LABEL_PATH = "./dataset/labels.npy"
LMDB_PATH  = "./dataset/patches.lmdb"

patches = np.load(PATCH_PATH, mmap_mode="r")
labels  = np.load(LABEL_PATH, mmap_mode="r")

env = lmdb.open(
    LMDB_PATH,
    map_size=1024**4,   # 1 TB max (safe large cap)
    subdir=False,
    meminit=False,
    map_async=True
)

with env.begin(write=True) as txn:
    txn.put(b"length", str(len(patches)).encode())

    for i in tqdm(range(len(patches))):
        data = {
            "patch": patches[i].astype(np.float16),  # optional compression
            "label": int(labels[i])
        }
        txn.put(str(i).encode(), pickle.dumps(data))

env.sync()
env.close()

print("✅ LMDB Conversion Complete")