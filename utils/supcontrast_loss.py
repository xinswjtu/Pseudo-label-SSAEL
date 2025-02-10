'''
supervised contrastive learning

reference:
[1] 《Supervised Contrastive Learning》 URL: https://github.com/HobbitLong/SupContrast
[2] 《Conditional Contrastive Domain Generalization For Fault Diagnosis》
[3] 《A Simple Framework for Contrastive Learning of Visual Representations》 https://arxiv.org/abs/2002.05709
'''
from torch import nn
import torch.nn.functional as F
import torch


class SupContrastLoss(nn.Module):
    def __init__(self, temperature, device):
        '''
        URL: https://github.com/mohamedr002/CCDG
        temperature (float): tradeoff
        '''
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, domains_features, domains_labels):
        '''
        Args:
            domains_features (tensor): (b, n_features)
            domains_labels (tensor): (b,)
        Returns:
                SupConstrastLoss (tensor): scaler
        '''
        # masking for the corresponding class labels.
        anchor_feature = domains_features
        anchor_feature = F.normalize(anchor_feature, dim=1) # normalize to sphere
        labels = domains_labels
        labels = labels.contiguous().view(-1, 1)
        # Generate masking for positive and negative pairs.
        mask = torch.eq(labels, labels.T).float().to(self.device)
        # Applying contrastive via product among the features
        # Measure the similarity between all samples in the batch
        # reiterate fact from Linear Algebra if u and v two vectors are normalised implies cosine similarity
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, anchor_feature.T), self.temperature)

        # for numerical stability
        # substract max value from the output
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # create inverted identity matrix with same shape as mask.
        # only diagnoal is zeros and all others are ones
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(anchor_feature.shape[0]).view(-1, 1).to(self.device),
                                    0)
        # mask-out self-contrast cases
        # the diagnoal represent same samples similarity i=j>> we need to remove this
        # remove the true labels mask
        # all ones in mask would be only samples with same labels
        mask = mask * logits_mask

        # compute log_prob and remove the diagnal
        # remove same features from the normalized contrastive matrix
        # The denominoator of the equation samples
        exp_logits = torch.exp(logits) * logits_mask

        # substract the whole multiplications from the negative pair and positive pairs
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mask_sum = mask.sum(1)
        zeros_idx = torch.where(mask_sum == 0)[0]
        mask_sum[zeros_idx] = 1

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # loss
        loss = (- 1 * mean_log_prob_pos)
        loss = loss.mean()
        return loss
