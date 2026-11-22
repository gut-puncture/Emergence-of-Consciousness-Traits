"""Write the savanna checkpoints the paper's results use, as safetensors, for the Hugging Face repo ect-savanna-runs.

    python prepare_runs.py <out_dir> <keep_dir>

keep_dir is the folder holding world_i1_seeds/, world_i1_stages23_seed0/ and world_i1_stages23_seeds1to4/.
The layout mirrors the training code's runs/ folder: s{stage}_seed{seed}/weights_{update}.safetensors (run k of the
paper is seed k-1). Weights only (no optimizer state), FP32; the output head is tied to the input embedding and
not stored twice. 23 files of about 401 MB each, about 9.2 GB.
"""
import os
import sys

# (stage, seed, update) -> source file under keep_dir
SOURCES = {}
for s in range(3):
    for u in (3000, 4000):
        SOURCES[(1, s, u)] = f"world_i1_seeds/seed{s}/weights_{u}.pt"
for s in (3, 4):   # weights_3000 of seeds 3 and 4 were not kept
    SOURCES[(1, s, 4000)] = f"world_i1_stages23_seeds1to4/s1_seed{s}/weights_4000.pt"
SOURCES[(2, 0, 1000)] = "world_i1_stages23_seed0/s2_seed0/weights_1000.pt"
for u in (1000, 1500):
    SOURCES[(3, 0, u)] = f"world_i1_stages23_seed0/s3_seed0/weights_{u}.pt"
for s in range(1, 5):
    SOURCES[(2, s, 1000)] = f"world_i1_stages23_seeds1to4/s2_seed{s}/weights_1000.pt"
    for u in (1000, 1500):
        SOURCES[(3, s, u)] = f"world_i1_stages23_seeds1to4/s3_seed{s}/weights_{u}.pt"


def prepare(out_dir, keep_dir, only=None):
    import torch
    from safetensors.torch import save_file
    for (stage, seed, u), src in sorted(SOURCES.items()):
        if only and (stage, seed, u) not in only:
            continue
        sd = torch.load(os.path.join(keep_dir, src), map_location="cpu", weights_only=True)
        assert torch.equal(sd["head.weight"], sd["embedding.weight"]) and len(sd) == 115, src
        sd = {k: v.float().contiguous() for k, v in sd.items() if k != "head.weight"}
        path = os.path.join(out_dir, f"s{stage}_seed{seed}", f"weights_{u}.safetensors")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_file(sd, path, metadata={"format": "pt"})
        print(path, os.path.getsize(path), flush=True)


if __name__ == "__main__":
    prepare(sys.argv[1], sys.argv[2])
