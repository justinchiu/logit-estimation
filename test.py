from pathlib import Path
import torch
from vec2text import analyze_utils
import datasets


vector_path = Path("saved_logits")
files = [x for x in vector_path.glob("*.npy") if x.is_file()]
example_idxs = set([int(str(f.stem).split("-")[0]) for f in files])

dataset = load_dataset("wentingzhao/one-million-instructions")["train"]


import pdb; pdb.set_trace()




experiment, trainer = analyze_utils.load_experiment_and_trainer_from_pretrained(
    "jxm/t5-base__llama-7b__one-million-paired-instructions",
)

trainer.suffix_ensemble = False
trainer.model.use_frozen_embeddings_as_input = True
trainer.args.per_device_eval_batch_size = 1

# vocab_size = trainer.embedder_tokenizer.vocab_size
vocab_size = 32768

trainer.generate(
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


