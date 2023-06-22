import torch



def PCloss(logits: torch.Tensor, tokens: torch.Tensor, use_ce=True):
    """
    :param logits: output word logits from the generation model in shape [N, L, E]
    :param tokens:
    :param use_ce:
    :return:
    """
    token_length = logits.shape[1]
    vocab_size = logits.shape[2]
    loss_weight = 100  
    fp_weights = [4*idx/token_length for idx in range(token_length)]
    fp_weights = torch.FloatTensor(fp_weights).to(logits.device)

    if use_ce:
        ce_thresh = 0.90
        fp_weights_ = fp_weights[None, fp_weights < ce_thresh, None]
        prob = nnf.sigmoid(logits[:, fp_weights < ce_thresh])
        label = nnf.one_hot(tokens[:, fp_weights < ce_thresh], num_classes=vocab_size)
        bce_entropy = label * torch.log(prob + EPSILON)
        bce_inv_entropy = (1 - label) * torch.log(1 - prob + EPSILON) * fp_weights_
        bce_loss = -torch.mean(bce_entropy+bce_inv_entropy)*loss_weight
        ce_loss = nnf.cross_entropy(logits[:, fp_weights >= ce_thresh].reshape(-1, logits.shape[-1]),
                                    tokens[:, fp_weights >= ce_thresh].flatten())
        loss = bce_loss + ce_loss
        return loss
    else:
        fp_weights = fp_weights[None, :, None]
        prob = nnf.sigmoid(logits)
        label = nnf.one_hot(tokens, num_classes=vocab_size)
        bce_entropy = label * torch.log(prob + EPSILON)
        bce_inv_entropy = (1 - label) * torch.log(1 - prob + EPSILON) * fp_weights
        loss = -torch.mean(bce_entropy+bce_inv_entropy)*loss_weight
        return loss
