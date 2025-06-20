
import torch

def gpt2_attention_mask_func(attention_scores, ltor_mask):
    mask_value = torch.finfo(attention_scores.dtype).min
    # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
    # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
    mask_value = torch.tensor(mask_value, dtype=attention_scores.dtype, device=attention_scores.device)
    attention_scores.masked_fill_(ltor_mask, mask_value)
    return attention_scores

def torch_softmax(input, mask):
    mask_output = gpt2_attention_mask_func(input, mask) 
    probs = torch.nn.Softmax(dim=-1)(mask_output)
    return probs