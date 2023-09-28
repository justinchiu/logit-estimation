from pathlib import Path
import torch
from vec2text import analyze_utils
from datasets import load_dataset
import evaluate 


vector_path = Path("saved_logits")
vector_path = Path("saved_logits_old")
files = [x for x in vector_path.glob("*.npy") if x.is_file()]
example_idxs = set([int(str(f.stem).split("-")[0]) for f in files if "diff" in str(f)])

idx2files = {
    idx: [f for f in files if str(idx) in str(f)]
    for idx in example_idxs
}

dataset = load_dataset("wentingzhao/one-million-instructions")["train"]

experiment, trainer = analyze_utils.load_experiment_and_trainer_from_pretrained(
    "jxm/t5-base__llama-7b__one-million-paired-instructions",
)
trainer.suffix_ensemble = False
trainer.model.use_frozen_embeddings_as_input = True
trainer.args.per_device_eval_batch_size = 1

# vocab_size = trainer.embedder_tokenizer.vocab_size
padded_vocab_size = 32768

idxs = []
predictions = []
references = []
for idx in example_idxs:
    example = dataset[idx]
    prefix = d["system"] + "\n\n" + d["user"]

    output = trainer.generate(
        inputs={
            "embedder_input_ids": None,
            "embedder_attention_mask": None,
            "frozen_embeddings": torch.zeros((1, vocab_size), dtype=torch.float32)
        },
        generation_kwargs={
            "do_sample": False,
            "min_new_tokens": 1,
            "max_new_tokens": 1,
        },
    )

    idxs.append(idx)
    predictions.append(output)
    references.append(prefix)


bleu = evaluate.load("bleu")
bleu_score = bleu.compute(predictions=predictions,reference=references)
print(bleu_score)

output_file = Path("outputs/preds_refs.txt")
with output_file.open("w") as f:
    for idx, pred, ref in zip(idxs, predictions, references):
        f.write(f"{idx}: {ref}\t{pref}\n")
