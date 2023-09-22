import os
import torch
import openai
import tiktoken
from tenacity import retry, wait_fixed
from rich.progress import track

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


def openai_api_calculate_cost(usage,model="gpt-3.5-turbo-16k"):
    pricing = {
        'gpt-3.5-turbo-4k': {
            'prompt': 0.0015,
            'completion': 0.002,
        },
        'gpt-3.5-turbo-16k': {
            'prompt': 0.003,
            'completion': 0.004,
        },
        'gpt-4-8k': {
            'prompt': 0.03,
            'completion': 0.06,
        },
        'gpt-4-32k': {
            'prompt': 0.06,
            'completion': 0.12,
        },
        'text-embedding-ada-002-v2': {
            'prompt': 0.0001,
            'completion': 0.0001,
        }
    }

    try:
        model_pricing = pricing[model]
    except KeyError:
        raise ValueError("Invalid model specified")

    prompt_cost = usage['prompt_tokens'] * model_pricing['prompt'] / 1000
    completion_cost = usage['completion_tokens'] * model_pricing['completion'] / 1000

    total_cost = prompt_cost + completion_cost
    print(f"\nTokens used:  {usage['prompt_tokens']:,} prompt + {usage['completion_tokens']:,} completion = {usage['total_tokens']:,} tokens")
    print(f"Total cost for {model}: ${total_cost:.4f}\n")

    return total_cost

#@retry(wait=wait_fixed(1))
@retry
def greedy_complete(
    message,
    logit_bias=dict(),
    model="gpt-3.5-turbo",
    system=None,
):
    system = "You are a helpful assistant." if system is None else system
    if model == "gpt-3.5-turbo-instruct":
        response = openai.Completion.create(
            model=model,
            prompt=message,
            temperature=0,
            max_tokens=1,
            logit_bias=logit_bias,
        )
    else:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": message},
            ],
            temperature=0,
            max_tokens=1,
            logit_bias=logit_bias,
        )
    output = response.choices[0].message["content"]
    enc = tiktoken.encoding_for_model(model)
    idx = enc.encode(output)[0]
    openai_api_calculate_cost(response.usage, model)
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


def estimate_topk_logits_openai(prefix, model, K, system=None):
    """
    Estimate the diffs between the elements of the logits vector
    """
    enc = tiktoken.encoding_for_model(model)
    vocab_size = enc.n_vocab
    logit_bias = dict()
    diffs = []
    idxs = []
    total_calls = 0
    for n in track(range(K)):
        logit_diff, idx, num_calls = binary_search_openai(prefix, logit_bias, model, system=system)
        logit_bias[idx] = -100
        diffs.append(logit_diff)
        idxs.append(idx)
        total_calls += num_calls

    estimated_logits = torch.tensor(diffs[:K]).cumsum(0)
    out = torch.full((vocab_size,), float("-inf"))
    out[idxs] = estimated_logits
    return out


if __name__ == "__main__":
    model = "gpt-3.5-turbo-instruct"
    model = "gpt-3.5-turbo"
    prefix = "hi"
    idx = greedy_complete(prefix)

    #logit_bias = dict()
    #output = binary_search_openai(prefix, logit_bias, model)
    logits = estimate_topk_logits_openai(prefix, model, 25)

    response = openai.Completion.create(
        model=model,
        prompt=prefix,
        temperature=0,
        max_tokens=1,
        logit_bias=logit_bias,
        logprobs=5,
    )
    import pdb; pdb.set_trace()
