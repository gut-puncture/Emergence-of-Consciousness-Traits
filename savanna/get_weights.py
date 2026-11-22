"""Download the released weights into the files the savanna code reads (this script is new; the rest of savanna/ is
the code as run).

    python get_weights.py                     # base_weights.pt and tokenizer.json, for stage 1
    python get_weights.py s3_seed0/weights_1500 s2_seed0/weights_1000 ...
                                              # also trained checkpoints, as runs/<name>.pt

The base model comes from AngryYoungBhaloo/ect-base-100m and the savanna checkpoints from
AngryYoungBhaloo/ect-savanna-runs (run k of the paper is seed k-1). Weights are FP32.
"""
import json, os, sys
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
BASE, RUNS = "AngryYoungBhaloo/ect-base-100m", "AngryYoungBhaloo/ect-savanna-runs"


def tied(sd):
    sd["head.weight"] = sd["embedding.weight"]   # the output head shares the input embedding
    return sd


def main(names):
    if not os.path.exists(os.path.join(HERE, "base_weights.pt")):
        config = json.load(open(hf_hub_download(BASE, "config.json")))
        torch.save(dict(config=config, weights=tied(load_file(hf_hub_download(BASE, "model.safetensors")))),
                   os.path.join(HERE, "base_weights.pt"))
    if not os.path.exists(os.path.join(HERE, "tokenizer.json")):
        with open(hf_hub_download(BASE, "tokenizer.json"), "rb") as f, open(os.path.join(HERE, "tokenizer.json"), "wb") as g:
            g.write(f.read())
    for name in names:
        out = os.path.join(HERE, "runs", name + ".pt")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        torch.save(tied(load_file(hf_hub_download(RUNS, name + ".safetensors"))), out)
        print(out)


if __name__ == "__main__":
    main(sys.argv[1:])
