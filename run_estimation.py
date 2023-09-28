import numpy as np
from datasets import load_dataset
from logit_estimation.estimators import naive_estimate, HfSampler, diffsearch
import random

N = 32000

dataset = load_dataset("wentingzhao/one-million-instructions")
index = random.randint(0, len(dataset))
d = dataset["train"][index]
prefix = f"[INST] <<SYS>>\n{d['system']}\n<</SYS>>\n {d['user']} [/INST]"

#model = "meta-llama/Llama-2-7b-chat-hf"
model = "meta-llama/Llama-2-7b-hf"
sampler = HfSampler(model)
np.save(f"saved_logits/{index}-true.npy", sampler.cached_logits.numpy())

estimator, num_calls = naive_estimate(sampler, prefix, N)
log_probs = estimator.mean()
np.save(f"saved_logits/{index}-mc.npy", log_probs)

idxs, estimated_logits, logit_bias, total_calls = diffsearch(sampler, prefix, N, dict())
lp = np.full((sampler.vocab_size,), float("-inf"), dtype=np.float64)
Zhat = logsumexp(estimated_logits)
lp[idxs] = estimated_logits - Zhat
np.save(f"saved_logits/{index}-diff.npy", lp)