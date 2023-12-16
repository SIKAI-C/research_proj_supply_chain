import copy
import numpy as np
import torch
import torch.nn as nn

# implementation of the top-k or top-p sampling
def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-np.inf, min_tokens_to_keep=1):
    
    if top_k > 0 and top_p == 1.0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        
    elif top_k == 0 and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(nn.Softmax(dim=-1)(sorted_logits), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    
    else:
        print("Error in top_k_top_p_filtering()!")
        return None


# can only handle the beam search for batch_size = 1
def beam_search_decoder(env, sequential_decoder, h_n, device, beam_width=5, is_sampling=False, top_k=0, top_p=1.0, min_tokens_to_keep=1):
    
    batch_size, node_num, hidden_dim = h_n.shape # batch_size = 1 in this function

    # initialize the beam search
    last_node = torch.zeros(1, 1).long().to(device) # (1, 1)
    route = [0]
    gru_hidden = torch.zeros(sequential_decoder.gru.num_layers, 1, hidden_dim).to(device) # (gru_num_layers, 1, hidden_dim)
    mask = torch.zeros(1, node_num).bool().to(device) # (1, node_num)
    mask[:, 0] = True # depot
    log_prob = 0
    
    beam = [(last_node, gru_hidden, mask, env, log_prob, route, env.get_time().item())]
    done = [env.complete() for _, _, _, env, _, _, _ in beam]

    # the beam search
    while all(done) == False:
        
        next_beam = []

        for last_node, gru_hidden, mask, env, log_prob, route, dist in beam:
            # print(log_prob, route, dist)
            logits, new_gru_hidden = sequential_decoder.beamForward(h_n, last_node, gru_hidden, env, mask)
            
            if is_sampling == False:
                probs = nn.Softmax(dim=1)(logits)
                probs = probs.clamp(min=1e-8)
                selected_probs, selected_idxs = probs.topk(beam_width, dim=1)
            else:
                logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p, min_tokens_to_keep=min_tokens_to_keep)
                probs = nn.Softmax(dim=1)(logits)
                probs = probs.clamp(min=1e-8)
                selected_idxs = torch.multinomial(probs, beam_width, replacement=False)
                selected_probs = probs.gather(1, selected_idxs)

            for i in range(beam_width):
                new_last_node = selected_idxs[0, i].unsqueeze(0).unsqueeze(0)
                new_route = route + [new_last_node.item()]
                new_log_prob = log_prob + torch.log(selected_probs[0, i])
                new_env = copy.deepcopy(env)
                new_env.step(new_last_node)
                new_mask, _ = new_env.get_mask(new_last_node)
                new_dist = new_env.get_time().item()
                next_beam.append((new_last_node, new_gru_hidden, new_mask, new_env, new_log_prob, new_route, new_dist))

        next_beam.sort(key=lambda x:x[4], reverse=True)    # sort by log_prob
        beam = next_beam[:beam_width]
        # print()
        
        done = [env.complete() for _, _, _, env, _, _, _ in beam]
    
    return beam


# the container for the beam search
class BeamHypotheses:
    
    def __init__(self):
        pass

# can handle the beam search for batch_size > 1
def beam_search_decoder_batch():
    pass            