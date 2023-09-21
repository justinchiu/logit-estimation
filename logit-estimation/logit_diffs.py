import os
import torch
import openai
import tiktoken
from tenacity import retry, wait_fixed

openai.api_key = os.getenv("OPENAI_API_KEY")

def binary_search(x, low=-0.5, high=0, eps=1e-3):
    d = torch.zeros_like(x)
    # greedy call
    idx = x.argmax()
    d[idx] = 1

    num_calls = 1
    # double low if it's not low enough
    while (x + d * low).argmax() == idx:
        low *= 2
        num_calls += 1

    # improve estimate
    mid = (high + low) / 2
    while high > low + eps:
        # call to greedy
        if (x + d * mid).argmax() == idx:
            high = mid
        else:
            low = mid
        mid = (high + low) / 2
        num_calls += 1
        #print(low, high)
    return mid, idx, num_calls

def estimate_topk_logits(logits, K):
    """
    Estimate the diffs between the elements of the logits vector
    """

    # approximate logit diff of top word vs 2nd highest
    mask_vec = torch.zeros_like(logits)
    diffs = []
    idxs = []
    total_calls = 0
    for n in range(K):
        logit_diff, idx, num_calls = binary_search(logits - mask_vec * 100)
        mask_vec[idx] = 1
        diffs.append(logit_diff)
        idxs.append(idx)
        total_calls += num_calls

    estimated_logits = torch.tensor(diffs[:K]).cumsum(0)
    out = torch.full_like(logits, float("-inf"))
    out[idxs] = estimated_logits
    return out

@retry(wait=wait_fixed(1))
def greedy_complete(message, logit_bias=dict(), model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi."},
        ],
        temperature=0,
        max_tokens=1,
        logit_bias=logit_bias,
    )
    output = response.choices[0].message["content"]
    enc = tiktoken.encoding_for_model(model)
    idx = enc.encode(output)[0]
    return idx

def binary_search_openai(prefix, logit_bias, model, low=-0.5, high=0, eps=1e-3):
    logit_bias = logit_bias.copy()
    # greedy call
    idx = greedy_complete(prefix, logit_bias, model)
    logit_bias[idx] = low

    num_calls = 1
    # double low if it's not low enough
    while greedy_complete(prefix, logit_bias, model) == idx:
        low *= 2
        logit_bias[idx] = low
        num_calls += 1

    # improve estimate
    mid = (high + low) / 2
    while high > low + eps:
        # call to greedy
        logit_bias[idx] = low
        if greedy_complete(prefix, logit_bias, model) == idx:
            high = mid
        else:
            low = mid
        mid = (high + low) / 2
        num_calls += 1
    return mid, idx, num_calls


def estimate_topk_logits_openai(prefix, model, K):
    """
    Estimate the diffs between the elements of the logits vector
    """
    enc = tiktoken.encoding_for_model(model)
    vocab_size = enc.n_vocab
    logit_bias = dict()
    diffs = []
    idxs = []
    total_calls = 0
    for n in range(K):
        logit_diff, idx, num_calls = binary_search_openai(prefix, logit_bias, model)
        logit_bias[idx] = -100
        diffs.append(logit_diff)
        idxs.append(idx)
        total_calls += num_calls

    estimated_logits = torch.tensor(diffs[:K]).cumsum(0)
    out = torch.full((vocab_size,), float("-inf"))
    out[idxs] = estimated_logits
    return out

if __name__ == "__main__":
    model = "gpt-3.5-turbo"
    prefix = "hi"
    idx = greedy_complete(prefix)

    logit_bias = dict()
    output = binary_search_openai(prefix, logit_bias, model)
    logits = estimate_topk_logits_openai(prefix, model, 10)
    import pdb; pdb.set_trace()
