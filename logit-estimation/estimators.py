from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.distributions import Categorical, kl_divergence


@dataclass
class Output:
    samples: list[int] | None = None
    argmax: int | None = None
    true_dist: torch.distributions.Distribution | None = None


class Sampler(ABC):
    @abstractmethod
    def sample(self, prefix, n, logit_bias, temperature):
        ...

def construct_logit_bias_tensor(logit_bias_dict, vocab_size):
    logit_bias = torch.zeros(vocab_size)
    for idx, value in logit_bias_dict.items():
        logit_bias[idx] = value
    return logit_bias

class HfSampler(Sampler):
    def __init__(self, model):
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.vocab_size = len(self.tokenizer)
        self.cached_logits = None

    def sample(self, prefix, K, logit_bias=None, temperature=1):
        logits = (
            self.get_true_logits(prefix, K)
            if self.cached_logits is None
            else self.cached_logits
        )
        if logit_bias is not None:
            logits = logits + construct_logit_bias_tensor(logit_bias, len(logits))

        true_dist = Categorical(logits=logits)
        return Output(
            samples = true_dist.sample_n(K),
            argmax = logits.argmax().item(),
            true_dist = true_dist,
        )

    def get_true_logits(self, prefix, K):
        inputs = self.tokenizer([prefix], return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=True,
            #num_return_sequences=1,
        )
        logits = outputs.scores[0][0].log_softmax(0)
        self.cached_logits = logits
        return logits


class Estimator:
    def __init__(self, vocab_size, threshold = 10, estimated_logits=None, idxs=None):
        self.vocab_size = vocab_size
        self.samples = []
        self.weights = []
        self.Zs = torch.zeros(vocab_size, dtype=torch.float64)
        self.counts = torch.zeros(vocab_size, dtype=torch.int32)
        self.threshold = threshold

        # already estimated
        self.estimated_logits = estimated_logits
        self.idxs = idxs

    def add_sample(self, sample_output, weight=1., allowed_words=None):
        for sample in sample_output.samples:
            self.samples.append(sample)
            self.weights.append(weight)
        self.counts.scatter_add_(
            0,
            sample_output.samples,
            torch.ones_like(sample_output.samples, dtype=torch.int32),
        )
        #Z = weight * len(sample_output.samples)
        Z = len(sample_output.samples)
        if allowed_words is not None:
            self.Zs[allowed_words] += Z
        else:
            self.Zs += Z

    def mean(self):
        if len(self.samples) == 0:
            return None
        probs = torch.zeros(self.vocab_size) + 1e-20 # smoothing
        probs.scatter_add_(
            0,
            torch.tensor(self.samples, dtype=torch.int64),
            torch.tensor(self.weights),
        )
        probs /= self.Zs

        if self.estimated_logits is not None:
            #Z1 = probs.log().logsumexp(0)
            #Z2 = self.estimated_logits.logsumexp(0)
            P = probs[self.idxs].sum()
            #import pdb; pdb.set_trace()
            #probs[self.idxs] = self.estimated_logits.softmax(0) * P
        return probs / probs.sum()

    def weight(self):
        mean = self.mean()
        words = self.confident_words()
        return 1. - mean[words].sum() if mean is not None else 1., words

    def confident_words(self):
        words = (self.counts > self.threshold).nonzero()[:,0].tolist()
        return words
        return (
            list(set(words + self.idxs))
            if self.idxs is not None
            else words
        )


def binary_search(sampler, prefix, logit_bias, low=-0.5, high=0, eps=1e-5):
    logit_bias = logit_bias.copy()
    idx = sampler.sample(prefix, 1, logit_bias, temperature=0).argmax
    logit_bias[idx] = low

    #print(sampler.cached_logits.topk(2).values[0] - sampler.cached_logits.topk(2).values[1])
    num_calls = 1
    # double low if it's not low enough
    while sampler.sample(prefix, 1, logit_bias, temperature=0).argmax == idx:
        low *= 2
        logit_bias[idx] = low
        num_calls += 1

    # improve estimate
    mid = (high + low) / 2
    while high > low + eps:
        logit_bias[idx] = mid
        if sampler.sample(prefix, 1, logit_bias, temperature=0).argmax == idx:
            high = mid
        else:
            low = mid
        mid = (high + low) / 2
        num_calls += 1
    return mid, idx, num_calls


def estimate(sampler, prefix, K, T, threshold):
    estimator = Estimator(sampler.vocab_size, threshold)
    for t in range(T):
        weight, words = estimator.weight()
        #logit_bias = {word: -100 for word in words}
        logit_bias = {word: -1000 for word in words}
        sample_output = sampler.sample(prefix, K, logit_bias)
        allowed_words = [x for x in range(sampler.vocab_size) if x not in logit_bias]
        estimator.add_sample(sample_output, weight, allowed_words=allowed_words)
    return estimator, K*T

def naive_estimate(sampler, prefix, K):
    estimator = Estimator(sampler.vocab_size)
    sample_output = sampler.sample(prefix, K, dict())
    estimator.add_sample(sample_output)
    return estimator, K

def search(sampler, prefix, topk, logit_bias, bias=-100):
    logit_bias = logit_bias.copy()

    diffs = []
    idxs = []
    total_calls = 0
    for _ in range(16):
        logit_diff, idx, num_calls = binary_search(sampler, prefix, logit_bias)
        logit_bias[idx] = bias
        diffs.append(logit_diff)
        idxs.append(idx)
        total_calls += num_calls

    estimated_logits = torch.tensor(diffs).cumsum(0)
    return idxs, estimated_logits, logit_bias, total_calls

def search_then_estimate(sampler, prefix, K, T, threshold):
    idxs, estimated_logits, logit_bias, total_calls = search(sampler, prefix, 16, dict())

    bias = -100
    estimator = Estimator(sampler.vocab_size, threshold,
        estimated_logits=estimated_logits, idxs=idxs)
    for t in range(T):
        weight, words = estimator.weight()
        logit_bias = {word: bias for word in words}
        sample_output = sampler.sample(prefix, K, logit_bias)
        allowed_words = [x for x in range(sampler.vocab_size) if x not in logit_bias]
        estimator.add_sample(sample_output, weight, allowed_words=allowed_words)
    return estimator, total_calls + K*T

if __name__ == "__main__":
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    llama = "meta-llama/Llama-2-7b-chat-hf"
    gpt = "gpt2"

    model = gpt
    if model == llama:
        prefix = "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\nHi [/INST]"
    else:
        prefix = "hi"


    # test sampling
    sampler = HfSampler(model)
    output = sampler.sample(prefix, 128)
    true_dist = output.true_dist

    #K = 1024
    K = 512
    tau = K // 2

    Ts = [2, 4, 8, 16]
    #Ts = [8, 16, 32, 64]

    method_list = []
    samples_list = []
    kl_list = []

    max_prob_list = []
    prob_5_list = []
    prob_10_list = []
    prob_20_list = []
    prob_30_list = []

    for _ in range(10):
        for T in Ts:
            # test estimation
            e1, c1 = estimate(sampler, prefix, K, T, tau)
            e2, c2 = naive_estimate(sampler, prefix, K*T)
            e3, c3 = search_then_estimate(sampler, prefix, K, T, tau)
            mu1 = e1.mean()
            mu2 = e2.mean()
            mu3 = e3.mean()
            kl1 = kl_divergence(true_dist, Categorical(probs=mu1)).item()
            kl2 = kl_divergence(true_dist, Categorical(probs=mu2)).item()
            kl3 = kl_divergence(true_dist, Categorical(probs=mu3)).item()

            method_list.append("Truncate sample")
            method_list.append("Naive sample")
            method_list.append("Search then sample")
            samples_list.append(c1)
            samples_list.append(c2)
            samples_list.append(c3)
            kl_list.append(kl1)
            kl_list.append(kl2)
            kl_list.append(kl3)

            max_prob_list.append(mu1.max().item())
            max_prob_list.append(mu2.max().item())
            max_prob_list.append(mu3.max().item())
            prob_5_list.append(mu1.topk(5).values[-1].item())
            prob_5_list.append(mu2.topk(5).values[-1].item())
            prob_5_list.append(mu3.topk(5).values[-1].item())
            prob_10_list.append(mu1.topk(10).values[-1].item())
            prob_10_list.append(mu2.topk(10).values[-1].item())
            prob_10_list.append(mu3.topk(10).values[-1].item())
            prob_20_list.append(mu1.topk(20).values[-1].item())
            prob_20_list.append(mu2.topk(20).values[-1].item())
            prob_20_list.append(mu3.topk(20).values[-1].item())
            prob_30_list.append(mu1.topk(30).values[-1].item())
            prob_30_list.append(mu2.topk(30).values[-1].item())
            prob_30_list.append(mu3.topk(30).values[-1].item())


    df = pd.DataFrame({
        "x": samples_list,
        "y": kl_list,
        "max": max_prob_list,
        "5": prob_5_list,
        "10": prob_10_list,
        "20": prob_20_list,
        "30": prob_30_list,
        "method": method_list,
    })
    sns.lineplot(data=df, x="x", y="y", hue="method", errorbar="sd")
    plt.title("Scatter Plot of Num Samples vs KL")
    plt.xlabel("Num samples")
    plt.ylabel("KL")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/truncated_samples_kl.png")
    plt.clf()


    sns.lineplot(data=df, x="x", y="max", hue="method", errorbar="sd")
    plt.axhline(y=true_dist.probs.max().item(), color="red", linestyle="--")
    plt.title("Scatter Plot of Num Samples vs max probability estimation")
    plt.xlabel("Num samples")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/truncated_samples_max.png")
    plt.clf()

    sns.lineplot(data=df, x="x", y="5", hue="method", errorbar="sd")
    plt.axhline(y=true_dist.probs.topk(5).values[-1].item(), color="red", linestyle="--")
    plt.title("Scatter Plot of Num Samples vs 5th rank probability estimation")
    plt.xlabel("Num samples")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/truncated_samples_5.png")
    plt.clf()

    sns.lineplot(data=df, x="x", y="10", hue="method", errorbar="sd")
    plt.axhline(y=true_dist.probs.topk(10).values[-1].item(), color="red", linestyle="--")
    plt.title("Scatter Plot of Num Samples vs 10th rank probability estimation")
    plt.xlabel("Num samples")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/truncated_samples_10.png")
    plt.clf()

    sns.lineplot(data=df, x="x", y="20", hue="method", errorbar="sd")
    plt.axhline(y=true_dist.probs.topk(20).values[-1].item(), color="red", linestyle="--")
    plt.title("Scatter Plot of Num Samples vs 20th rank probability estimation")
    plt.xlabel("Num samples")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/truncated_samples_20.png")
    plt.clf()

    sns.lineplot(data=df, x="x", y="30", hue="method", errorbar="sd")
    plt.axhline(y=true_dist.probs.topk(30).values[-1].item(), color="red", linestyle="--")
    plt.title("Scatter Plot of Num Samples vs 30th rank probability estimation")
    plt.xlabel("Num samples")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/truncated_samples_30.png")
    plt.clf()
