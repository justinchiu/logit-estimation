from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from torch.distributions import Categorical, kl_divergence

torch.set_grad_enabled(False)

llama = "meta-llama/Llama-2-7b-chat-hf"
gpt = "gpt2"

gpt_tokenizer = AutoTokenizer.from_pretrained(gpt)
llama_tokenizer = AutoTokenizer.from_pretrained(llama)
t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")

gpt_model = AutoModelForCausalLM.from_pretrained(gpt)
# llama_model = AutoModelForCausalLM.from_pretrained(llama)


def get_kl(prefix, K):
    inputs = gpt_tokenizer([prefix], return_tensors="pt")
    outputs = gpt_model.generate(
        **inputs,
        max_new_tokens=2,
        return_dict_in_generate=True,
        output_scores=True,
        do_sample=True,
        num_return_sequences=K,
    )

    true_dist = Categorical(logits=outputs.scores[0][0])
    observations = outputs.sequences[:, 1]

    V = len(true_dist.enumerate_support())
    counts = torch.zeros(V, dtype=torch.int32)
    counts = counts.scatter_add(
        0, observations, torch.ones_like(observations, dtype=torch.int32)
    )
    sample_estimate = Categorical(probs=counts + 1e-5)  # counts.log().log_softmax(0)
    sample_kl = kl_divergence(true_dist, sample_estimate)

    topk = true_dist.logits.topk(5)
    # needs a lot of smoothing
    logits = torch.full((V,), -10, dtype=torch.float32)
    logits[topk.indices] = topk.values
    topk_estimate = Categorical(logits=logits)
    topk_kl = kl_divergence(true_dist, topk_estimate)

    topk_mass = topk.values.logsumexp(0).exp()
    sample_logits = sample_estimate.logits + 0
    sample_logits[topk.indices] = float("-inf")
    sample_mass = sample_logits.logsumexp(0).exp()
    sample_probs = sample_estimate.probs / sample_mass * (1 - topk_mass)
    sample_probs[topk.indices] = topk.values.exp()

    sample_topk_kl = kl_divergence(true_dist, Categorical(probs=sample_probs),)

    return sample_kl, topk_kl, sample_topk_kl


prefix = "Hi"
kls = []
methods = []
Ks = [8, 16, 32, 64, 128, 256, 512]
Ks_df = []
for K in Ks:
    sample_kl, topk_kl, sample_topk_kl = get_kl(prefix, K)
    Ks_df.append(K)
    kls.append(sample_kl)
    methods.append("sample")

    Ks_df.append(K)
    kls.append(topk_kl)
    methods.append("top5")

    Ks_df.append(K)
    kls.append(sample_topk_kl)
    methods.append("sample+top5")

df = pd.DataFrame({"x": Ks_df, "y": kls, "method": methods})
sns.scatterplot(data=df, x="x", y="y", hue="method")

plt.title("Scatter Plot of Num Samples vs KL")
plt.xlabel("Num samples")
plt.ylabel("KL")
plt.legend()
plt.tight_layout()
plt.savefig("figures/samples_kl.png")
# import pdb; pdb.set_trace()

# use logit bias and greedy?
