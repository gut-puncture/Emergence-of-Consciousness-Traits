"""Prepare and upload the three Hugging Face repositories of the paper.

    python upload.py WORK_DIR --keep KEEP_DIR [--prepare-only]

WORK_DIR must be outside the synced notes vault: the large files (about 15 GB) are written there, next to copies of
the cards in this folder, and uploaded from there. Each large file is written only if it is not there yet, so the
command can be run again after an interruption; upload_large_folder resumes too.

  AngryYoungBhaloo/ect-base-100m           model    model.safetensors (401 MB), config.json, tokenizer.json, card
  AngryYoungBhaloo/ect-savanna-runs        model    23 checkpoints (9.2 GB), card
  AngryYoungBhaloo/ect-pretraining-corpus  dataset  43 parquet shards (5.0 GB), parts.json, manifest/, labels/, prepare_corpus.py, card
"""
import argparse
import os
import shutil

import prepare_base
import prepare_corpus
import prepare_runs

HERE = os.path.dirname(os.path.abspath(__file__))
OWNER = "AngryYoungBhaloo"
REPOS = {"ect-base-100m": "model", "ect-savanna-runs": "model", "ect-pretraining-corpus": "dataset"}


def prepare(work, keep):
    base = os.path.join(work, "ect-base-100m")
    if not os.path.exists(os.path.join(base, "model.safetensors")):
        prepare_base.prepare(base, keep)
    runs = os.path.join(work, "ect-savanna-runs")
    missing = {k for k in prepare_runs.SOURCES
               if not os.path.exists(os.path.join(runs, f"s{k[0]}_seed{k[1]}", f"weights_{k[2]}.safetensors"))}
    if missing:
        prepare_runs.prepare(runs, keep, only=missing)
    corpus = os.path.join(work, "ect-pretraining-corpus")
    accepted = os.path.join(keep, "pretraining_data_a0_verified_core", "accepted")
    if not os.path.exists(os.path.join(corpus, "parts.json")):
        prepare_corpus.prepare(corpus, accepted)
    os.makedirs(os.path.join(corpus, "manifest"), exist_ok=True)
    for f in os.listdir(accepted):   # the original manifest and receipts: counts and hashes, no text
        if f.endswith(".json"):
            shutil.copy(os.path.join(accepted, f), os.path.join(corpus, "manifest"))
    shutil.copy(os.path.join(HERE, "prepare_corpus.py"), corpus)
    for repo in REPOS:   # the cards and small files, kept beside this script in the staging folder
        if os.path.isdir(os.path.join(HERE, repo)):
            shutil.copytree(os.path.join(HERE, repo), os.path.join(work, repo), dirs_exist_ok=True)


def upload(work):
    from huggingface_hub import HfApi
    api = HfApi()
    for repo, kind in REPOS.items():
        rid = f"{OWNER}/{repo}"
        api.create_repo(rid, repo_type=kind, private=False, exist_ok=True)
        api.upload_large_folder(repo_id=rid, repo_type=kind, folder_path=os.path.join(work, repo))
        print("uploaded", rid, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--keep", required=True, help="the folder holding the kept run records and data")
    ap.add_argument("--prepare-only", action="store_true")
    a = ap.parse_args()
    vault = os.path.dirname(os.path.dirname(HERE))
    assert os.path.commonpath([vault, os.path.abspath(a.work)]) != vault, "write the large files outside the notes vault"
    prepare(a.work, a.keep)
    if not a.prepare_only:
        upload(a.work)
