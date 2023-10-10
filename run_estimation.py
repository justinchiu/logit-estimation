import numpy as np
from datasets import load_dataset
from logit_estimation.estimators import naive_estimate, HfSampler, batch_diffsearch, diffsearch
import random
from scipy.special import logsumexp

from pathlib import Path

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("index", type=int)
parser.add_argument("eps", type=float, default=1e-8)
args = parser.parse_args()

index = args.index
eps = args.eps

Path(f"saved_logits-batch").mkdir(exist_ok=True)

dataset = load_dataset("wentingzhao/one-million-instructions")["train"]
#index = random.randint(0, len(dataset))
d = dataset[index]
#prefix = f"[INST] <<SYS>>\n{d['system']}\n<</SYS>>\n {d['user']} [/INST]"
prefix = d["system"] + "\n\n" + d["user"]

#model = "meta-llama/Llama-2-7b-chat-hf"
model = "meta-llama/Llama-2-7b-hf"
sampler = HfSampler(model)
sampler.sample(prefix, 1)
np.save(f"saved_logits-batch/{index}-true.npy", sampler.cached_logits.numpy())

batch_lp, total_calls = batch_diffsearch(sampler, prefix, eps=eps)
np.save(f"saved_logits-batch/{index}-batch-diff-{total_calls}-eps{eps}.npy", batch_lp)

estimator, num_calls = naive_estimate(sampler, prefix, total_calls)
log_probs = estimator.mean()
np.save(f"saved_logits-batch/{index}-mc-{total_calls}.npy", log_probs)

lp, total_calls = diffsearch(sampler, prefix, eps=eps)
np.save(f"saved_logits-batch/{index}-diff-{total_calls}-eps{eps}.npy", lp)

estimator, num_calls = naive_estimate(sampler, prefix, total_calls)
log_probs = estimator.mean()
np.save(f"saved_logits-batch/{index}-mc-{total_calls}.npy", log_probs)

