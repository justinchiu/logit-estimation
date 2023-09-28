from abc import ABC, abstractmethod
from dataclasses import dataclass
import sys
import math
import torch
import numpy as np
from scipy.special import logsumexp

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.distributions import Categorical, kl_divergence


np.seterr(all='raise')

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
        #self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.vocab_size = len(self.tokenizer)
        self.cached_logits = None
        self.cached_prefix = None

    def sample(self, prefix, K, logit_bias=None, temperature=1):
        if prefix != self.cached_prefix:
            self.cached_logits = None
            self.cached_prefix = prefix

        logits = (
            self.get_true_logits(prefix, K)
            if self.cached_logits is None
            else self.cached_logits
        )
        if logit_bias is not None:
            logits = logits + construct_logit_bias_tensor(logit_bias, len(logits))

        true_dist = Categorical(logits=logits)

        return Output(
            samples = true_dist.sample_n(K).tolist() if temperature > 0 else None,
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
            num_return_sequences=1,
            top_p=1.0,
            top_k=0,
            num_beams=1,
        )
        #import pdb; pdb.set_trace()
        logits = outputs.scores[0][0].log_softmax(0)
        self.cached_logits = logits
        return logits

class GptSampler(Sampler):
    def __init__(self, model):
        #self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16)
        self.model = model
        self.vocab_size = 100000

    def sample(self, prefix, K, logit_bias=None, temperature=1, system=None):
        model = self.model
        system = "You are a helpful assistant." if system is None else system

        enc = tiktoken.encoding_for_model(model)
        if model == "gpt-3.5-turbo-instruct":
            response = openai.Completion.create(
                model=model,
                prompt=prefix,
                temperature=temperature,
                max_tokens=1,
                logit_bias=logit_bias,
                n=K,
            )
            output = response.choices[0].text
            eos_idx = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>", "<|im_start|>"})[0]
            outputs = [choice.text for choice in response.choices]
        else:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prefix},
                ],
                temperature=temperature,
                max_tokens=1,
                logit_bias=logit_bias,
                n=K,
            )
            output = response.choices[0].message["content"]
            outputs = [choice.message["content"] for choice in response.choices]
            eos_idx = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>", "<|im_start|>"})[0]

        if response.choices[0].finish_reason == "length":
            idx = enc.encode(output)[0]
        elif response.choices[0].finish_reason == "stop":
            idx = eos_idx
        else:
            import pdb; pdb.set_trace()

        if temperature > 0:
            samples = [enc.encode(output) for output in outputs]

        return Output(
            samples = samples if temperature > 0 else None,
            argmax = idx if temperature == 0 else None,
            true_dist = None,
        )


class Estimator:
    def __init__(self, vocab_size, threshold = 50, estimated_logits=None, idxs=None):
        self.vocab_size = vocab_size
        self.samples = []
        self.log_weights = []

        self.Zs = np.zeros(vocab_size, dtype=np.float64)
        self.counts = np.zeros(vocab_size, dtype=np.int64)
        self.threshold = threshold

        # already estimated
        self.estimated_logits = estimated_logits
        self.idxs = idxs

    def add_sample(self, sample_output, log_weight=0., allowed_words=None):
        for sample in sample_output.samples:
            self.samples.append(sample)
            self.log_weights.append(log_weight)
        np.add.at(self.counts, sample_output.samples, 1)
        Z = len(sample_output.samples)
        if allowed_words is not None:
            self.Zs[allowed_words] += Z
        else:
            self.Zs += Z

    def mean(self):
        if len(self.samples) == 0:
            return None

        s = np.array(self.samples, dtype=np.int64)
        w = np.array(self.log_weights, dtype=np.float64)

        lp = np.full((self.vocab_size,), float("-inf"), dtype=np.float64)
        #lp = np.full((self.vocab_size,), -sys.float_info.max, dtype=np.float64)
        #lp = np.full((self.vocab_size,), -1e12, dtype=np.float64)
        #lp = np.full((self.vocab_size,), -1e2, dtype=np.float64)
        np.logaddexp.at(lp, s, w)
        # is this numerically stable?
         
        # try a really slow version
        #import pdb; pdb.set_trace()

        log_probs = lp - np.log(self.Zs)

        if self.estimated_logits is not None:
            Zsample = logsumexp(log_probs[self.idxs])
            Zhat = logsumexp(self.estimated_logits)
            log_probs[self.idxs] = self.estimated_logits - Zhat + Zsample
        return log_probs

    def weight(self):
        mean = self.mean()
        words = self.confident_words()
        if mean is not None and len(words) > 0:
            log_mass = logsumexp(mean[words])
            if log_mass >= 0:
                return None, words
            return math.log(1 - math.exp(log_mass)), words
        else:
            return 0, words

    def confident_words(self):
        words = (self.counts > self.threshold).nonzero()[0].tolist()
        return words

def query_ordering(sampler, prefix, bias=-1000):
    vocab_size = sampler.vocab_size
    logit_bias = dict()
    ordering = []
    for i in range(vocab_size):
        idx = sampler.sample(prefix, 1, logit_bias, temperature=0).argmax
        logit_bias[idx] = bias
        ordering.append(idx)
    return ordering

def binary_search(sampler, prefix, logit_bias, low=-0.25, high=0, eps=1e-10):
    logit_bias = logit_bias.copy()
    idx = sampler.sample(prefix, 1, logit_bias, temperature=0).argmax
    logit_bias[idx] = low
    idx_lower = None

    #print(sampler.cached_logits.topk(2).values[0] - sampler.cached_logits.topk(2).values[1])
    num_calls = 1
    # double low if it's not low enough
    while sampler.sample(prefix, 1, logit_bias, temperature=0).argmax == idx:
        low *= 2
        logit_bias[idx] = low
        num_calls += 1
        if low < -1e3:
            # likely a -inf for the next word
            import pdb; pdb.set_trace()
            return float("-inf", idx, num_calls)

    # improve estimate
    mid = (high + low) / 2
    while high > low + eps:
        logit_bias[idx] = mid
        idx2 = sampler.sample(prefix, 1, logit_bias, temperature=0).argmax
        if idx2 == idx:
            high = mid
        else:
            low = mid
            idx_lower = idx2
        mid = (high + low) / 2
        num_calls += 1
    return mid, idx, num_calls, idx_lower


def estimate(sampler, prefix, K, T, threshold):
    estimator = Estimator(sampler.vocab_size, threshold)
    for t in range(T):
        weight, words = estimator.weight()
        if weight is None:
            break
        #logit_bias = {word: -100 for word in words}
        logit_bias = {word: -1000 for word in words}
        sample_output = sampler.sample(prefix, K, logit_bias)
        allowed_words = [x for x in range(sampler.vocab_size) if x not in logit_bias]
        estimator.add_sample(sample_output, weight, allowed_words=allowed_words)
    return estimator, K*(t+1)

def naive_estimate(sampler, prefix, K):
    estimator = Estimator(sampler.vocab_size)
    sample_output = sampler.sample(prefix, K, dict())
    estimator.add_sample(sample_output)
    return estimator, K

def search(sampler, prefix, topk, logit_bias, bias=-100):
    logit_bias = logit_bias.copy()
    highest_idx = sampler.sample(prefix, 1, logit_bias, temperature=0).argmax

    diffs = [0]
    idxs = [highest_idx]
    total_calls = 0
    for _ in range(topk):
        logit_diff, idx, num_calls, lower_idx = binary_search(sampler, prefix, logit_bias)
        logit_bias[idx] = bias
        diffs.append(logit_diff)
        idxs.append(lower_idx)
        total_calls += num_calls
        print(total_calls, _, topk)

    estimated_logits = np.array(diffs, dtype=np.float64).cumsum()
    return idxs, estimated_logits, logit_bias, total_calls

def diffsearch(sampler, prefix, topk, logit_bias, bias=-1000):
    logit_bias = logit_bias.copy()
    highest_idx = sampler.sample(prefix, 1, logit_bias, temperature=0).argmax

    diffs = [0]
    idxs = [highest_idx]
    total_calls = 1
    logit_diff = 0
    for _ in range(topk):
        logit_diff, idx, num_calls, idx_lower = binary_search(sampler, prefix, logit_bias, high=logit_diff)
        logit_bias[idx_lower] = bias
        diffs.append(logit_diff)
        idxs.append(idx_lower)
        total_calls += num_calls
        print(total_calls, _, topk)

    estimated_logits = np.array(diffs, dtype=np.float64)
    return idxs, estimated_logits, logit_bias, total_calls

def search_then_estimate(sampler, prefix, K, T, threshold):
    #idxs, estimated_logits, logit_bias, total_calls = search(sampler, prefix, 16, dict())
    #idxs, estimated_logits, logit_bias, total_calls = search(sampler, prefix, 64, dict())
    idxs, estimated_logits, logit_bias, total_calls = diffsearch(sampler, prefix, 128, dict())

    bias = -1000
    estimator = Estimator(sampler.vocab_size, threshold,
        estimated_logits=estimated_logits, idxs=idxs)

    remaining_calls = K*T - total_calls
    newT = remaining_calls // K
    for t in range(newT):
        print(t, newT)
        weight, words = estimator.weight()
        if weight is None:
            break
        logit_bias = {word: bias for word in words}
        sample_output = sampler.sample(prefix, K, logit_bias)
        allowed_words = [x for x in range(sampler.vocab_size) if x not in logit_bias]
        estimator.add_sample(sample_output, weight, allowed_words=allowed_words)

    weight, words = estimator.weight()
    logit_bias = {word: bias for word in words}
    if weight is not None:
        sample_output = sampler.sample(prefix, remaining_calls % K, logit_bias)
        allowed_words = [x for x in range(sampler.vocab_size) if x not in logit_bias]
        estimator.add_sample(sample_output, weight, allowed_words=allowed_words)

    return estimator, K*T

def search_then_sample(sampler, prefix, K, T, threshold):
    idxs, estimated_logits, logit_bias, total_calls = diffsearch(sampler, prefix, 128, dict())

    bias = -1000
    estimator = Estimator(sampler.vocab_size, threshold,
        estimated_logits=estimated_logits, idxs=idxs)

    remaining_calls = K*T - total_calls
    sample_output = sampler.sample(prefix, remaining_calls)
    estimator.add_sample(sample_output)
    return estimator, K*T


if __name__ == "__main__":
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    llama = "meta-llama/Llama-2-7b-chat-hf"
    #llama = "meta-llama/Llama-2-7b-hf"
    gpt = "gpt2"

    USE_LLAMA = False
    USE_LLAMA = True

    if not USE_LLAMA:
        model = gpt
        model_name = "gpt"
    else:
        model = llama
        model_name = "llama"
    if model == llama:
        from datasets import load_dataset
        dataset = load_dataset("wentingzhao/one-million-instructions")
        d = dataset["train"][0]
        prefix = f"[INST] <<SYS>>\n{d['system']}\n<</SYS>>\n {d['user']} [/INST]"
    else:
        prefix = "hi"



    # test sampling
    sampler = HfSampler(model)
    output = sampler.sample(prefix, 128)
    true_dist = output.true_dist


    """
    encoded_prompt = sampler.tokenizer(prefix, return_tensors="pt")

    out = sampler.model.generate(
          input_ids=encoded_prompt["input_ids"],
          attention_mask=encoded_prompt["attention_mask"],
          do_sample=False,
          max_length=64 + len(encoded_prompt[0]),
    )

    import pdb; pdb.set_trace()
    """

    """
    idxs1, estimated_logits1, logit_bias1, total_calls1 = diffsearch(sampler, prefix, 2**14, dict())
    idxs2, estimated_logits2, logit_bias2, total_calls2 = search(sampler, prefix, 2**14, dict())
    import pdb; pdb.set_trace()
    """

    K = 2**14
    K = 2**10
    tau = 2*K

    Ts = [32,64,128, 256]
    Ts = [16,32,64]

    method_list = []
    samples_list = []
    kl_list = []
    rmse_list = []
    rrmse_list = []

    max_prob_list = []
    prob_25_list = []
    prob_50_list = []
    prob_100_list = []
    prob_500_list = []
    prob_1000_list = []

    for _ in range(10):
        for T in Ts:
            # test estimation
            #e1, c1 = estimate(sampler, prefix, K, T, tau)
            e1, c1 = search_then_sample(sampler, prefix, K, T, tau)
            e2, c2 = naive_estimate(sampler, prefix, K*T)
            e3, c3 = search_then_estimate(sampler, prefix, K, T, tau)
            mu1 = torch.tensor(np.exp(e1.mean()))
            mu2 = torch.tensor(np.exp(e2.mean()))
            mu3 = torch.tensor(np.exp(e3.mean()))
            kl1 = kl_divergence(true_dist, Categorical(probs=mu1)).item()
            kl2 = kl_divergence(true_dist, Categorical(probs=mu2)).item()
            kl3 = kl_divergence(true_dist, Categorical(probs=mu3)).item()
            rmse1 = (true_dist.probs - mu1).square().mean().sqrt().item()
            rmse2 = (true_dist.probs - mu2).square().mean().sqrt().item()
            rmse3 = (true_dist.probs - mu3).square().mean().sqrt().item()
            rrmse1 = ((true_dist.probs - mu1).abs() / true_dist.probs).mean().item()
            rrmse2 = ((true_dist.probs - mu2).abs() / true_dist.probs).mean().item()
            rrmse3 = ((true_dist.probs - mu3).abs() / true_dist.probs).mean().item()

            #method_list.append("Truncate sample")
            method_list.append("Search then sample")
            method_list.append("Naive sample")
            method_list.append("Search then truncate sample")
            samples_list.append(c1)
            samples_list.append(c2)
            samples_list.append(c3)
            kl_list.append(kl1)
            kl_list.append(kl2)
            kl_list.append(kl3)
            rmse_list.append(rmse1)
            rmse_list.append(rmse2)
            rmse_list.append(rmse3)
            rrmse_list.append(rrmse1)
            rrmse_list.append(rrmse2)
            rrmse_list.append(rrmse3)
            #if kl1 < 0 or kl2 < 0 or kl3 < 0:
            #    print(kl1, kl2, kl3)

            max_prob_list.append(mu1.max().item())
            max_prob_list.append(mu2.max().item())
            max_prob_list.append(mu3.max().item())
            prob_25_list.append(mu1.topk(25).values[-1].item())
            prob_25_list.append(mu2.topk(25).values[-1].item())
            prob_25_list.append(mu3.topk(25).values[-1].item())
            prob_50_list.append(mu1.topk(50).values[-1].item())
            prob_50_list.append(mu2.topk(50).values[-1].item())
            prob_50_list.append(mu3.topk(50).values[-1].item())
            prob_100_list.append(mu1.topk(100).values[-1].item())
            prob_100_list.append(mu2.topk(100).values[-1].item())
            prob_100_list.append(mu3.topk(100).values[-1].item())
            prob_500_list.append(mu1.topk(500).values[-1].item())
            prob_500_list.append(mu2.topk(500).values[-1].item())
            prob_500_list.append(mu3.topk(500).values[-1].item())
            prob_1000_list.append(mu1.topk(1000).values[-1].item())
            prob_1000_list.append(mu2.topk(1000).values[-1].item())
            prob_1000_list.append(mu3.topk(1000).values[-1].item())


    df = pd.DataFrame({
        "x": samples_list,
        "kl": kl_list,
        "rmse": rmse_list,
        "rrmse": rrmse_list,
        "max": max_prob_list,
        "25": prob_25_list,
        "50": prob_50_list,
        "100": prob_100_list,
        "500": prob_500_list,
        "1000": prob_1000_list,
        "method": method_list,
    })
    sns.lineplot(data=df, x="x", y="kl", hue="method", errorbar="sd")
    plt.title("Scatter Plot of Num Samples vs KL")
    plt.xlabel("Num samples")
    plt.ylabel("KL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}_truncated_samples_kl.png")
    plt.clf()

    sns.lineplot(data=df, x="x", y="rmse", hue="method", errorbar="sd")
    plt.title("Scatter Plot of Num Samples vs RMSE")
    plt.xlabel("Num samples")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}_truncated_samples_rmse.png")
    plt.clf()

    sns.lineplot(data=df, x="x", y="rrmse", hue="method", errorbar="sd")
    plt.title("Scatter Plot of Num Samples vs Relative RMSE")
    plt.xlabel("Num samples")
    plt.ylabel("Relative RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}_truncated_samples_rrmse.png")
    plt.clf()


    sns.lineplot(data=df, x="x", y="max", hue="method", errorbar="sd")
    plt.axhline(y=true_dist.probs.max().item(), color="red", linestyle="--")
    plt.title("Scatter Plot of Num Samples vs max probability estimation")
    plt.xlabel("Num samples")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}_truncated_samples_max.png")
    plt.clf()

    sns.lineplot(data=df, x="x", y="25", hue="method", errorbar="sd")
    plt.axhline(y=true_dist.probs.topk(25).values[-1].item(), color="red", linestyle="--")
    plt.title("Scatter Plot of Num Samples vs 25th rank probability estimation")
    plt.xlabel("Num samples")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}_truncated_samples_25.png")
    plt.clf()

    sns.lineplot(data=df, x="x", y="50", hue="method", errorbar="sd")
    plt.axhline(y=true_dist.probs.topk(50).values[-1].item(), color="red", linestyle="--")
    plt.title("Scatter Plot of Num Samples vs 50th rank probability estimation")
    plt.xlabel("Num samples")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}_truncated_samples_50.png")
    plt.clf()

    sns.lineplot(data=df, x="x", y="100", hue="method", errorbar="sd")
    plt.axhline(y=true_dist.probs.topk(100).values[-1].item(), color="red", linestyle="--")
    plt.title("Scatter Plot of Num Samples vs 100th rank probability estimation")
    plt.xlabel("Num samples")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}_truncated_samples_100.png")
    plt.clf()

    sns.lineplot(data=df, x="x", y="500", hue="method", errorbar="sd")
    plt.axhline(y=true_dist.probs.topk(500).values[-1].item(), color="red", linestyle="--")
    plt.title("Scatter Plot of Num Samples vs 500th rank probability estimation")
    plt.xlabel("Num samples")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}_truncated_samples_500.png")
    plt.clf()

    sns.lineplot(data=df, x="x", y="1000", hue="method", errorbar="sd")
    plt.axhline(y=true_dist.probs.topk(1000).values[-1].item(), color="red", linestyle="--")
    plt.title("Scatter Plot of Num Samples vs 1000th rank probability estimation")
    plt.xlabel("Num samples")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}_truncated_samples_1000.png")
    plt.clf()
