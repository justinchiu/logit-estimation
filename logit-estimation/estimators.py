from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.distributions import Categorical, kl_divergence


@dataclass
class Output:
    samples: list[int]
    true_dist: torch.distributions.Distribution | None


class Sampler(ABC):
    @abstractmethod
    def sample(self, prefix, n, logit_bias):
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

    def sample(self, prefix, K, logit_bias=None):
        inputs = self.tokenizer([prefix], return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=True,
            #num_return_sequences=1,
        )

        logits = outputs.scores[0][0]
        if logit_bias is not None:
            logits = logits + construct_logit_bias_tensor(logit_bias, len(logits))
        true_dist = Categorical(logits=logits)
        return Output(
            samples = true_dist.sample_n(K),
            true_dist = true_dist,
        )


class Estimator:
    def __init__(self, vocab_size, threshold = 10):
        self.vocab_size = vocab_size
        self.samples = []
        self.weights = []
        self.Zs = torch.zeros(vocab_size, dtype=torch.float64)
        self.counts = torch.zeros(vocab_size, dtype=torch.int32)
        self.threshold = threshold

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
        probs = torch.zeros(self.vocab_size) + 1e-75 # smoothing
        probs.scatter_add_(
            0,
            torch.tensor(self.samples, dtype=torch.int64),
            torch.tensor(self.weights),
        )
        probs /= self.Zs
        return probs / probs.sum()

    def weight(self):
        mean = self.mean()
        words = self.confident_words()
        return 1. - mean[words].sum() if mean is not None else 1., words

    def confident_words(self):
        return (self.counts > self.threshold).nonzero()[:,0].tolist()

def estimate(sampler, prefix, K, T, threshold):
    estimator = Estimator(sampler.vocab_size, threshold)
    for t in range(T):
        weight, words = estimator.weight()
        #logit_bias = {word: -100 for word in words}
        logit_bias = {word: -1000 for word in words}
        sample_output = sampler.sample(prefix, K, logit_bias)
        allowed_words = [x for x in range(sampler.vocab_size) if x not in logit_bias]
        estimator.add_sample(sample_output, weight, allowed_words=allowed_words)
    return estimator

def naive_estimate(sampler, prefix, K):
    estimator = Estimator(sampler.vocab_size)
    sample_output = sampler.sample(prefix, K, dict())
    estimator.add_sample(sample_output)
    return estimator


if __name__ == "__main__":
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    llama = "meta-llama/Llama-2-7b-chat-hf"
    gpt = "gpt2"
    prefix = "hi"

    # test sampling
    sampler = HfSampler(gpt)
    output = sampler.sample(prefix, 128)
    true_dist = output.true_dist

    #K = 1024
    K = 512
    tau = K // 2

    #Ts = [2, 4, 8, 16, 32]
    Ts = [8, 16, 32, 64]

    method_list = []
    samples_list = []
    kl_list = []

    max_prob_list = []
    prob_25_list = []
    prob_50_list = []
    prob_75_list = []

    for _ in range(10):
        for T in Ts:
            # test estimation
            e1 = estimate(sampler, prefix, K, T, tau)
            e2 = naive_estimate(sampler, prefix, K*T)
            mu1 = e1.mean()
            mu2 = e2.mean()
            kl1 = kl_divergence(true_dist, Categorical(probs=mu1)).item()
            kl2 = kl_divergence(true_dist, Categorical(probs=mu2)).item()

            method_list.append("Truncate sample")
            method_list.append("Naive sample")
            samples_list.append(K*T)
            samples_list.append(K*T)
            kl_list.append(kl1)
            kl_list.append(kl2)

            max_prob_list.append(mu1.max().item())
            max_prob_list.append(mu2.max().item())
            prob_25_list.append(mu1.topk(25).values[-1].item())
            prob_25_list.append(mu2.topk(25).values[-1].item())
            prob_50_list.append(mu1.topk(50).values[-1].item())
            prob_50_list.append(mu2.topk(50).values[-1].item())
            prob_75_list.append(mu1.topk(75).values[-1].item())
            prob_75_list.append(mu2.topk(75).values[-1].item())


    df = pd.DataFrame({
        "x": samples_list,
        "y": kl_list,
        "max": max_prob_list,
        "25": prob_25_list,
        "50": prob_50_list,
        "75": prob_75_list,
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

    sns.lineplot(data=df, x="x", y="25", hue="method", errorbar="sd")
    plt.axhline(y=true_dist.probs.topk(25).values[-1].item(), color="red", linestyle="--")
    plt.title("Scatter Plot of Num Samples vs 25th rank probability estimation")
    plt.xlabel("Num samples")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/truncated_samples_25.png")
    plt.clf()

    sns.lineplot(data=df, x="x", y="50", hue="method", errorbar="sd")
    plt.axhline(y=true_dist.probs.topk(50).values[-1].item(), color="red", linestyle="--")
    plt.title("Scatter Plot of Num Samples vs 50th rank probability estimation")
    plt.xlabel("Num samples")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/truncated_samples_50.png")
    plt.clf()

    sns.lineplot(data=df, x="x", y="75", hue="method", errorbar="sd")
    plt.axhline(y=true_dist.probs.topk(75).values[-1].item(), color="red", linestyle="--")
    plt.title("Scatter Plot of Num Samples vs 75th rank probability estimation")
    plt.xlabel("Num samples")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/truncated_samples_75.png")
    plt.clf()
