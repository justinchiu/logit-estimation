import torch
from vec2text import analyze_utils

experiment, trainer = analyze_utils.load_experiment_and_trainer_from_pretrained(
    "jxm/t5-base__llama-7b__one-million-paired-instructions",
)

trainer.suffix_ensemble = False
trainer.model.use_frozen_embeddings_as_input = True
trainer.args.per_device_eval_batch_size = 1
#trainer.evaluate(
#    eval_dataset=trainer.eval_dataset["one_million_instructions"].select(range(10))
#)


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
