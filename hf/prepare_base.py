"""Write the base model's weights for the Hugging Face folder ect-base-100m/.

    python prepare_base.py <out_dir> <drive_path>

drive_path is the folder holding base_model_production_v3/ (the kept run of the original pre-training). The
final checkpoint (step 9,979) holds the weights and the optimizer state; only the weights are written, in
float32, as model.safetensors. The output head shares its weights with the input embedding, so it is not
stored twice (savanna/get_weights.py restores the tie). The file is about 401 MB: write it outside synced folders and
copy it next to README.md, config.json and tokenizer.json only when uploading.
"""
import hashlib
import os
import sys

CHECKPOINT = "base_model_production_v3/run/checkpoints/generation-00000052/state.pt"


def prepare(out_dir, drive_path):
    import torch
    from safetensors.torch import save_file
    state = torch.load(os.path.join(drive_path, CHECKPOINT), map_location="cpu", weights_only=True)["state"]
    assert state["step"] == 9_979 and state["consumed_loss_tokens"] == 2_610_181_035
    weights = state["model"]
    assert torch.equal(weights["head.weight"], weights["embedding.weight"])
    weights = {k: v.float().contiguous() for k, v in weights.items() if k != "head.weight"}
    assert sum(v.numel() for v in weights.values()) == 100_291_200
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "model.safetensors")
    save_file(weights, path, metadata={"format": "pt"})
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    print(f"{path}: {os.path.getsize(path):,} bytes, sha256 {h.hexdigest()}")
    return path


if __name__ == "__main__":
    prepare(sys.argv[1], sys.argv[2])
