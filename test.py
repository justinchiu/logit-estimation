from pathlib import Path
import torch
from vec2text import analyze_utils
from datasets import load_dataset
import evaluate
import numpy as np
from rich.progress import track


precision = "1e-8"

vector_path = Path("saved_logits-32000")
files = {
    "08": [
        "0-true.npy", 
        "0-mc-1199045.npy",
        "0-diff-1199045-eps1e-08.npy",
    ],
    "18": [
        "1-true.npy", 
        "1-mc-1200501.npy",
        "1-diff-1200501-eps1e-08.npy",
    ],
    "28": [
        "2-true.npy", 
        "2-mc-1198224.npy",
        "2-diff-1198224-eps1e-08.npy",
    ],
    "38": [
        "3-true.npy", 
        "3-mc-1196708.npy",
        "3-diff-1196708-eps1e-08.npy",
    ],
    "48": [
        "4-true.npy", 
        "4-mc-1205012.npy",
        "4-diff-1205012-eps1e-08.npy",
    ],
    "06": [
        "0-true.npy", 
        "0-mc-985629.npy",
        "0-diff-985629-eps1e-06.npy",
    ],
    "16": [
        "1-true.npy", 
        "1-mc-986990.npy",
        "1-diff-986990-eps1e-06.npy",
    ],
    "26": [
        "2-true.npy", 
        "2-mc-985016.npy",
        "2-diff-985016-eps1e-06.npy",
    ],
    "36": [
        "3-true.npy", 
        "3-mc-983562.npy",
        "3-diff-983562-eps1e-06.npy",
    ],
    "46": [
        "4-true.npy", 
        "4-mc-990877.npy",
        "4-diff-990877-eps1e-06.npy",
    ],
    "04": [
        "0-true.npy", 
        "0-mc-774355.npy",
        "0-diff-774355-eps0.0001.npy",
    ],
    "14": [
        "1-true.npy", 
        "1-mc-775259.npy",
        "1-diff-775259-eps0.0001.npy",
    ],
    "24": [
        "2-true.npy", 
        "2-mc-773567.npy",
        "2-diff-773567-eps0.0001.npy",
    ],
    "34": [
        "3-true.npy", 
        "3-mc-771699.npy",
        "3-diff-771699-eps0.0001.npy",
    ],
    "44": [
        "4-true.npy", 
        "4-mc-779349.npy",
        "4-diff-779349-eps0.0001.npy",
    ],
}

# validate files
for _, fs in files.items():
    for f in fs:
        assert (vector_path / f).is_file(), f

example_idxs = set([
    int(str(f.stem).split("-")[0])
    for f in files
    if "diff" in str(f) and precision in str(f)
])
example_idxs = [0,1,2,3,4]
precisions = [8,6,4]


dataset = load_dataset("wentingzhao/one-million-instructions")["train"]

experiment, trainer = analyze_utils.load_experiment_and_trainer_from_pretrained(
    "jxm/t5-base__llama-7b__one-million-paired-instructions",
)
trainer.suffix_ensemble = False
trainer.model.use_frozen_embeddings_as_input = True
trainer.args.per_device_eval_batch_size = 1

tokenizer = trainer.tokenizer

# vocab_size = trainer.embedder_tokenizer.vocab_size
padded_vocab_size = 32768

for precision in precisions:
    idxs_diff = []
    predictions_diff = []
    references_diff = []
    idxs_mc = []
    predictions_mc = []
    references_mc = []
    idxs = []
    predictions = []
    references = []

    for example_idx in track(example_idxs):
        example = dataset[example_idx]
        prefix = example["system"] + "\n\n" + example["user"]
        for file in files[f"{example_idx}{precision}"]:
            f = str(vector_path / file)
            logprobsnp = np.load(f)
            logprobs = torch.zeros((1, padded_vocab_size), dtype=torch.float32)
            logprobs[0,:len(logprobsnp)] = torch.tensor(logprobsnp)
            if "diff" in f:
                output = trainer.generate(
                    inputs={
                        "embedder_input_ids": None,
                        "embedder_attention_mask": None,
                        "frozen_embeddings": logprobs,
                    },
                    generation_kwargs={
                        "do_sample": False,
                        "min_new_tokens": 1,
                        "max_new_tokens": 64,
                    },
                )
                idxs_diff.append(example_idx)
                predictions_diff.append(tokenizer.batch_decode(output)[0])
                references_diff.append([prefix])
            elif "mc" in f:
                output = trainer.generate(
                    inputs={
                        "embedder_input_ids": None,
                        "embedder_attention_mask": None,
                        "frozen_embeddings": logprobs
                    },
                    generation_kwargs={
                        "do_sample": False,
                        "min_new_tokens": 1,
                        "max_new_tokens": 64,
                    },
                )
                idxs_mc.append(example_idx)
                predictions_mc.append(tokenizer.batch_decode(output)[0])
                references_mc.append([prefix])
            else:
                output = trainer.generate(
                    inputs={
                        "embedder_input_ids": None,
                        "embedder_attention_mask": None,
                        "frozen_embeddings": logprobs
                    },
                    generation_kwargs={
                        "do_sample": False,
                        "min_new_tokens": 1,
                        "max_new_tokens": 64,
                    },
                )
                idxs.append(example_idx)
                predictions.append(tokenizer.batch_decode(output)[0])
                references.append([prefix])

    print("PRECISION @", precision)
    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(predictions=predictions,references=references)
    print("GT", bleu_score)
    bleu_score = bleu.compute(predictions=predictions_diff,references=references_diff)
    print("Diff", bleu_score)
    bleu_score = bleu.compute(predictions=predictions_mc,references=references_mc)
    print("MC", bleu_score)

    output_file = Path("outputs/preds_refs_gt.txt")
    with output_file.open("w") as f:
        for idx, pred, ref in zip(idxs, predictions, references):
            f.write(f"{idx}: {ref}\t{pred}\n")
    output_file = Path(f"outputs/preds_refs_diff_{precision}.txt")
    with output_file.open("w") as f:
        for idx, pred, ref in zip(idxs_diff, predictions_diff, references_diff):
            f.write(f"{idx}: {ref}\t{pred}\n")
    output_file = Path("outputs/preds_refs_mc.txt")
    with output_file.open("w") as f:
        for idx, pred, ref in zip(idxs_mc, predictions_mc, references_mc):
            f.write(f"{idx}: {ref}\t{pred}\n")
