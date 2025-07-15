import torch
from torch import nn
from torch.nn import functional as F

import utils.config as config


def convert_sigmoid_logits_to_binary_logprobs(logits):
    """Computes log(sigmoid(logits)), log(1-sigmoid(logits))."""
    log_prob = -F.softplus(-logits)
    log_one_minus_prob = -logits + log_prob
    return log_prob, log_one_minus_prob


def cross_entropy_loss(logits, labels, **kwargs):
    """ Modified cross entropy loss. """
 
    nll = F.log_softmax(logits, dim=-1)
    loss = -nll * labels
    loss = loss 
    return loss.sum(dim=-1).mean()


def cross_entropy_loss_arc(logits, labels, **kwargs):
    """ Modified cross entropy loss. """
    f = kwargs['per']
    # d = kwargs['diff']
    # warm = kwargs['warm']
    nll = F.log_softmax(logits, dim=-1)
    loss = -nll * labels * f 
    # if warm:
    #    loss *= d.unsqueeze(1).expand_as(f)

    return loss.sum(dim=-1).mean()

class Plain(nn.Module):
    def forward(self, logits, labels, **kwargs):
        if config.loss_type == 'ce':
            loss = cross_entropy_loss(logits, labels, **kwargs)
        elif config.loss_type == 'ce_margin':
            loss = cross_entropy_loss_arc(logits, labels, **kwargs)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss *= labels.size(1)
        return loss



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, weights=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))



        batch_size = features.shape[0]
        
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_feature = features
        anchor_feature = contrast_feature

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        if weights is not None:
            logits = logits * weights

        # tile mask
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) # [bsz, bsz] - [bsz, 1]

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()

        return loss