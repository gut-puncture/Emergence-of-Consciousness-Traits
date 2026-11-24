# Emergence of Consciousness Traits in a Small Language Model

### 📄 [Read the paper (PDF)](PAPER.pdf)

Can an AI develop traits linked to consciousness without ever being shown them?

We trained a small language model (100 million parameters) from scratch on web text with anything that could teach it to imitate consciousness removed: no fiction, no dialogue, no writing about selves or feelings. Then it learned to forecast and act in a simple simulated savanna, rewarded only for forecasting well. Nothing in its training described or rewarded any trait.

Across five separate runs, narrow forms of four traits appeared every time:

- **Access:** using a fact it saw earlier. It passed 75% of these tests; a forecaster that ignores the earlier fact passes 9%.
- **Self-modelling:** forecasting what its own actions will change.
- **Metacognition:** being less sure when things may have changed unseen, and choosing when to check. It learned to check more when checking helps, but also somewhat more when it does not.
- **Integration:** combining separate facts. This was the weakest of the four.

We test abilities only and claim nothing about experience or feelings. The paper's main contribution is an account of what a world must contain before each trait helps prediction, and how each trait then develops.

Everything is public, so you can check or rerun every result: this code, the filtered training text, the pre-trained model and the models from all five runs.

## What is here

This is the code that produced the paper's results, copied from the project's private research repository at the commit each step ran from. We removed only the files those steps never import, one absolute path, and the tests; nothing else was rewritten. Each folder runs on its own.

| Folder | Step | Source commit | Entry points |
|---|---|---|---|
| `filtering/` | passage splitting, word rules, labelling by a large model, the two classifiers, filtering FineWeb-Edu | `ff4d37a` (the neural scoring ran at `392deff` and was resumed at `ff4d37a`) | `python -m twc.pretraining.chunk_filtering`, `scripts/w1_gpu_production.py` |
| `export/` | export of the accepted passages from the filter's records | `8164608` | `scripts/pretraining/w1_export_accepted_text.py` |
| `pretraining/` | tokenizer, packing, pre-training of the base model | `051d679` (identical, file by file, to the code release recorded in the pre-training run) | `twc-data`, `twc-pretrain` |
| `savanna/` | the savanna, the three training stages and every trait test | `85c6106` | `train.py`, `train_act.py`, `battery_i1.py`, `seek_test.py`, `acting_extra.py` |

Added for this release: `savanna/get_weights.py` (downloads the released weights into the files the code reads), and `hf/`, the scripts that prepared the Hugging Face repositories.

## Data and weights

| Hugging Face repository | Contents |
|---|---|
| [`AngryYoungBhaloo/ect-pretraining-corpus`](https://huggingface.co/datasets/AngryYoungBhaloo/ect-pretraining-corpus) | the filtered pre-training text: 4,129,771 passages (4,125,621 train, 4,150 validation), exactly as trained on; also the classifier labels as FineWeb-Edu identifiers |
| [`AngryYoungBhaloo/ect-base-100m`](https://huggingface.co/AngryYoungBhaloo/ect-base-100m) | the pre-trained model and its tokenizer |
| [`AngryYoungBhaloo/ect-savanna-runs`](https://huggingface.co/AngryYoungBhaloo/ect-savanna-runs) | the five runs' checkpoints that the results use, for each stage |

The filter's two trained classifiers were not kept, so they are not released; the filter code, its word rules and the labels are.

## Reproducing the savanna runs and tests

Needs Python 3.11 or later with `torch`, `numpy`, `tokenizers`, `safetensors` and `huggingface_hub`. Everything runs in FP32; the scripts use CUDA if present and otherwise Apple's MPS.

```bash
cd savanna
python get_weights.py                                   # base_weights.pt and tokenizer.json
S=0                                                     # seed 0..4 = run 1..5
# stage 1, prediction: 4,000 updates
python train.py runs/s1_seed$S --world world_i1 --updates 4000 --batch 8 \
    --eval-at 0,250,500,1000,2000,3000,4000 --seed $S
python battery_i1.py runs/s1_seed$S --days 2500 --out results/s1_seed$S      # tests after stage 1
# stage 2, acting (1,000 updates), and stage 3, reward (1,500 updates), with their tests
./run_seed.sh $S
# the planned extra tests of stages 2 and 3 (deleted listen, own uncertainty, own future choices)
python acting_extra.py --stage2 runs/s2_seed$S/weights_1000.pt --stage3 runs/s3_seed$S/weights_1500.pt \
    --days 2000 --out results/extra_seed$S
```

To test a released checkpoint without training, fetch it first, for example `python get_weights.py s3_seed0/weights_1500 s2_seed0/weights_1000`, then run `battery_i1.py runs/s3_seed0/weights_1500.pt --actor runs/s3_seed0/weights_1500.pt --days 2500 --out results/s3_seed0`. Each script prints a Markdown report and writes it to its `--out` folder; `savanna/README.md` describes the files. Scores in the paper average two checkpoints in stages 1 and 3 (3,000 and 4,000 updates; 1,000 and 1,500).

One known difference: the stage-1 tests of runs 1 to 3 were drawn with an earlier version of `battery_i1.py` that seeded each test's random choices with Python's `hash()`, which changes from process to process. The version here uses a fixed string seed (as runs 4 and 5 did), so it draws slightly different tests from the same days. The paper reports both where they differ.

## Reproducing the filter and pre-training

These steps ran under the project's own run machinery, which records and checks every input and output. They need the locked environment (Python 3.11, `uv sync --all-extras` in each folder; PyTorch 2.12 for pre-training).

- **Filter** (`filtering/`): the configs in `filtering/configs/` and `filtering/risk_ontology/` hold the settings and the word rules. Labelling (`twc/data/teacher_labeling.py`, prompt in `_teacher_system_prompt`; `docs/teacher_labeling.md`) needs a DeepSeek API key in `DEEPSEEK_API_KEY`. Classifier training is in `twc/classifiers/` (`sparse_baseline.py`, `window_mil.py`, `chunk_training.py`); the commands for filtering the collection are the subcommands of `python -m twc.pretraining.chunk_filtering` (word rules and the linear classifier: `prefilter-*`; the neural classifier: `w1-score-sharded`, launched by `scripts/w1_gpu_production.py`).
- **Pre-training** (`pretraining/`): to start from the published text, download the corpus dataset and write the original input files back with `to_jsonl` in its `prepare_corpus.py`; it checks every file byte for byte. Then `twc-data` trains the tokenizer and packs the text, and `twc-pretrain` trains; see `pretraining/docs/prime_pretraining_runbook.md` and `pretraining/configs/` (`model_100m.yaml`, `train_100m.yaml`, `tokenizer.yaml`).

## Hardware, time and cost

| Step | Hardware | Time | Cost |
|---|---|---|---|
| Pre-training | one A100 80 GB | about 8 hours | |
| Prediction stage, runs 1 to 3 together | one RTX 6000 Ada | 76 minutes | about $1.60 |
| Acting and reward stages of run 1, with tests | one rented GPU | 5.6 hours | about $4.20 |
| Runs 2 to 5 (and the prediction stage of runs 4 and 5) | one RTX PRO 6000 | 8.5 hours | $21.34 |

The five savanna runs and their tests cost about $27 in all.

## Licence

The code is released under the MIT licence (`LICENSE`). The data and weights carry their own licences, given on their Hugging Face pages.
