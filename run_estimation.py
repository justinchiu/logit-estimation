import numpy as np
from datasets import load_dataset
from logit_estimation.estimators import naive_estimate, HfSampler, diffsearch
import random
from scipy.special import logsumexp

from pathlib import Path

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("index", type=int)
parser.add_argument("N", type=int)
parser.add_argument("eps", type=float, default=1e-8)
args = parser.parse_args()

index = args.index
N = args.N
eps = args.eps
#N = 32000

Path(f"saved_logits-{N}").mkdir(exist_ok=True)

dataset = load_dataset("wentingzhao/one-million-instructions")["train"]
#index = random.randint(0, len(dataset))
d = dataset[index]
#prefix = f"[INST] <<SYS>>\n{d['system']}\n<</SYS>>\n {d['user']} [/INST]"
prefix = d["system"] + "\n\n" + d["user"]

#model = "meta-llama/Llama-2-7b-chat-hf"
model = "meta-llama/Llama-2-7b-hf"
model = "gpt2"
sampler = HfSampler(model)
sampler.sample(prefix, 1)
np.save(f"saved_logits-{N}/{index}-true.npy", sampler.cached_logits.numpy())

idxs, estimated_logits, logit_bias, total_calls = diffsearch(sampler, prefix, N, eps=eps)
lp = np.full((sampler.vocab_size,), float("-inf"), dtype=np.float64)
Zhat = logsumexp(estimated_logits)
lp[idxs] = estimated_logits - Zhat
np.save(f"saved_logits-{N}/{index}-diff-{total_calls}-eps{eps}.npy", lp)

estimator, num_calls = naive_estimate(sampler, prefix, total_calls)
log_probs = estimator.mean()
np.save(f"saved_logits-{N}/{index}-mc-{total_calls}.npy", log_probs)
