import numpy as np
from datasets import load_dataset
from logit_estimation.estimators import naive_estimate, GptSampler, gptdiffsearch
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

Path(f"saved_logits-gpt").mkdir(exist_ok=True)

dataset = load_dataset("wentingzhao/one-million-instructions")["train"]
#index = random.randint(0, len(dataset))
d = dataset[index]
#prefix = f"[INST] <<SYS>>\n{d['system']}\n<</SYS>>\n {d['user']} [/INST]"
#prefix = d["system"] + "\n\n" + d["user"]
prefix = d["user"]

model = "gpt-3.5-turbo-instruct"
sampler = GptSampler(model)

lp, total_calls = gptdiffsearch(sampler, prefix, eps=eps)
np.save(f"saved_logits-gpt/{index}-diff-{total_calls}-eps{eps}.npy", lp)

estimator, num_calls = naive_estimate(sampler, prefix, total_calls)
log_probs = estimator.mean()
np.save(f"saved_logits-gpt/{index}-mc-{total_calls}.npy", log_probs)

