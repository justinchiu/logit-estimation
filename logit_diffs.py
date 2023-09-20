import torch

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

def estimate_diffs(logits, K):
    """
    Estimate the diffs between the elements of the logits vector
    """
    softmax_top5 = logits.softmax(0).topk(5)
    top5 = logits.topk(5)

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

    true_logit_diff12 = top5.values[0] - top5.values[1]
    true_logit_diff23 = top5.values[1] - top5.values[2]
    true_logit_diff34 = top5.values[2] - top5.values[3]

    print(diffs[0], true_logit_diff12)
    print(diffs[1], true_logit_diff23)
    print(diffs[2], true_logit_diff34)
    print("total calls", total_calls)

    estimated_logits = torch.tensor(diffs[:K]).cumsum(0)
    out = torch.full_like(logits, float("-inf"))
    out[idxs] = estimated_logits
    return out
