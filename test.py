from pathlib import Path
import torch
from vec2text import analyze_utils
from datasets import load_dataset
import evaluate
import numpy as np
from rich.progress import track


vector_path = Path("saved_logits-batch")

example_idxs = list(range(100))
all_files = [f for f in vector_path.iterdir()]
files = {
    idx: [f for f in all_files if str(idx) == f.stem.split("-")[0]]
    for idx in example_idxs
}

# validate files
for _, fs in files.items():
    for f in fs:
        assert f.is_file(), f
#precisions = [6,5,4,3,2,1]
precisions = [6,4,3,2,1]

dataset = load_dataset("wentingzhao/one-million-instructions")["train"]

'''
experiment, trainer = analyze_utils.load_experiment_and_trainer_from_pretrained(
    "jxm/t5-base__llama-7b__one-million-paired-instructions",
)
trainer.suffix_ensemble = False
trainer.model.use_frozen_embeddings_as_input = True
trainer.args.per_device_eval_batch_size = 1

tokenizer = trainer.tokenizer
'''

# vocab_size = trainer.embedder_tokenizer.vocab_size
padded_vocab_size = 32768

    # batch-diff
    idxs_batch_diff = []
    predictions_batch_diff = []
    references_batch_diff = []
    calls_batch_diff = []
    # diff
    idxs_diff = []
    predictions_diff = []
    references_diff = []
    calls_diff = []
    # mc
    idxs_mc = []
    predictions_mc = []
    references_mc = []
    calls_mc = []
    # true
    idxs = []
    predictions = []
    references = []
    calls_true = []

    #for example_idx in track(example_idxs):
    for example_idx in example_idxs:
        example = dataset[example_idx]
        prefix = example["system"] + "\n\n" + example["user"]
        for file in files[example_idx]:
            f = str(file)
            import pdb; pdb.set_trace()
            logprobsnp = np.load(f)
            logprobsnp[logprobsnp < -100] = -30
            logprobs = torch.zeros((1, padded_vocab_size), dtype=torch.float32)
            logprobs[0,:len(logprobsnp)] = torch.tensor(logprobsnp)
            if "batch-diff" in f:
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
                idxs_batch_diff.append(example_idx)
                predictions_batch_diff.append(tokenizer.batch_decode(output)[0])
                references_batch_diff.append([prefix])
                calls_batch_diff.append()
            elif "diff" in f:
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
                calls_diff.append()
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
                calls_mc.append()
            elif "true" in f:
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
                calls_true.append(0)
            else:
                raise ValueError

    print("PRECISION @", precision)
    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(predictions=predictions,references=references)
    print("GT", bleu_score)
    bleu_score = bleu.compute(predictions=predictions_diff,references=references_diff)
    print("Diff", bleu_score)
    bleu_score = bleu.compute(predictions=predictions_batch_diff,references=references_batch_diff)
    print("Batch Diff", bleu_score)
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
    output_file = Path(f"outputs/preds_refs_batch_diff_{precision}.txt")
    with output_file.open("w") as f:
        for idx, pred, ref in zip(idxs_diff, predictions_batch_diff, references_diff):
            f.write(f"{idx}: {ref}\t{pred}\n")
    output_file = Path("outputs/preds_refs_mc.txt")
    with output_file.open("w") as f:
        for idx, pred, ref in zip(idxs_mc, predictions_mc, references_mc):
            f.write(f"{idx}: {ref}\t{pred}\n")
