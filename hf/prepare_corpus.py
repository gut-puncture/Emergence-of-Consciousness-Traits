"""Write the filtered pre-training corpus as parquet shards for the Hugging Face dataset ect-pretraining-corpus.

    python prepare_corpus.py <out_dir> <accepted_dir>

accepted_dir holds the 413 accepted_clean_runs-NNNNNN.jsonl parts the base model was trained on (4,129,771
deduplicated passages, sorted by text hash, 10,000 per part) and their receipts. Every field of every record is
kept, plus passage_index, the record's position in that order (0 .. 4,129,770). Train and validation rows go to
separate shards; ten original parts make one train shard. manifest/ keeps the original manifest and receipts
(counts and hashes only). parts.json lists each original part's SHA-256, so
`to_jsonl` (below) can rebuild the original files byte for byte. The output is about 6 GB.
"""
import glob
import json
import os
import shutil
import sys

PARTS_PER_SHARD = 10
COLUMNS = ["passage_index", "text", "source_document_id_hash", "split", "clean_run_id_hash", "clean_run_text_hash",
           "selected_block_count", "selected_block_root", "utf8_byte_count"]


def prepare(out_dir, accepted_dir):
    import pyarrow as pa
    import pyarrow.parquet as pq
    parts = sorted(glob.glob(os.path.join(accepted_dir, "accepted_clean_runs-*.jsonl")))
    assert len(parts) == 413, len(parts)
    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)
    index, val, listing, counts = 0, [], [], {"train": 0, "validation": 0}
    for s in range(0, len(parts), PARTS_PER_SHARD):
        train = []
        for p in parts[s:s + PARTS_PER_SHARD]:
            receipt = json.load(open(p[:-len(".jsonl")] + ".receipt.json"))
            n = 0
            with open(p, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    r["passage_index"] = index
                    (train if r["split"] == "train" else val).append(r)
                    counts[r["split"]] += 1
                    index += 1; n += 1
            assert n == receipt["record_count"], p
            listing.append(dict(part=os.path.basename(p), records=n, sha256=receipt["sha256"],
                                split_counts=receipt["split_counts"]))
        name = f"data/train-{s // PARTS_PER_SHARD:05d}-of-{-(-len(parts) // PARTS_PER_SHARD):05d}.parquet"
        pq.write_table(pa.Table.from_pylist(train).select(COLUMNS), os.path.join(out_dir, name), compression="zstd")
        print(name, len(train), flush=True)
    pq.write_table(pa.Table.from_pylist(val).select(COLUMNS),
                   os.path.join(out_dir, "data/validation-00000-of-00001.parquet"), compression="zstd")
    json.dump(dict(records=index, split_counts=counts, parts=listing), open(os.path.join(out_dir, "parts.json"), "w"),
              indent=1)
    # the original manifest and per-part receipts (counts and hashes, no text), which the packing step reads
    os.makedirs(os.path.join(out_dir, "manifest"), exist_ok=True)
    for f in glob.glob(os.path.join(accepted_dir, "*.json")):
        shutil.copy(f, os.path.join(out_dir, "manifest"))
    print("records", index, counts)
    assert index == 4_129_771 and counts == {"train": 4_125_621, "validation": 4_150}, (index, counts)


def to_jsonl(dataset_dir, out_dir):
    """The original accepted_clean_runs-NNNNNN.jsonl parts from the parquet shards, checked against parts.json."""
    import hashlib
    import pyarrow.parquet as pq
    listing = json.load(open(os.path.join(dataset_dir, "parts.json")))["parts"]
    val = {r["passage_index"]: r for f in glob.glob(os.path.join(dataset_dir, "data", "validation-*.parquet"))
           for r in pq.read_table(f).to_pylist()}

    def train_rows():
        for f in sorted(glob.glob(os.path.join(dataset_dir, "data", "train-*.parquet"))):
            yield from pq.read_table(f).to_pylist()

    train = train_rows()
    os.makedirs(out_dir, exist_ok=True)
    i = 0
    for part in listing:
        data = []
        for j in range(i, i + part["records"]):
            r = val.pop(j) if j in val else next(train)
            assert r.pop("passage_index") == j
            data.append(json.dumps(r, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        data = b"".join(data)
        i += part["records"]
        assert hashlib.sha256(data).hexdigest() == part["sha256"], part["part"]
        open(os.path.join(out_dir, part["part"]), "wb").write(data)
    for f in glob.glob(os.path.join(dataset_dir, "manifest", "*.json")):
        shutil.copy(f, out_dir)
    print(f"{len(listing)} parts, {i} records, all identical to the originals")


if __name__ == "__main__":
    prepare(sys.argv[1], sys.argv[2])
