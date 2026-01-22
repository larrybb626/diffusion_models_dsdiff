# -*- coding: utf-8 -*-
"""
@Project ：diffusion_models 
@File    ：contrastive_loss.py
@IDE     ：PyCharm 
@Author  ：MJY
@Date    ：2024/8/27 16:34 
"""
import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.1, contrastive_method='simclr'):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        self.contrastive_method = contrastive_method

    def _cosine_simililarity(self, x, y):
        """
        Shape:
        :param x: (N, 1, C)
        :param y: (1, N, C)
        :return: (N, N)
        """
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, features, labels=None, mask=None, temperature=None):
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
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            labels = torch.cat(torch.unbind(labels, dim=1), dim=0)
            labels = labels.contiguous().view(-1, 1).to(device)
            if self.contrastive_method == 'cl':
                mask = torch.eq(labels, labels.T).float().to(device)
            elif self.contrastive_method == 'infoNCE':
                mask = torch.eq(labels, labels.T).float().to(device)
                c_f = torch.cat(torch.unbind(features, dim=1), dim=0)
                labels = labels.view(-1)
                max_value = torch.max(labels)
                neg_values = labels[labels < 0]
                unique_neg_values = torch.unique(neg_values).sort(descending=True)[0]
                replacement_values = torch.arange(max_value + 1, max_value + 1 + len(unique_neg_values))
                for original, replacement in zip(unique_neg_values, replacement_values):
                    labels[labels == original] = replacement
                loss = nn.functional.cross_entropy(c_f, labels)
                # return loss
        else:
            mask = mask.float().to(device)
        # 创建一个perfect_logit，该矩阵对应mask为1的部分为1，mask为0的部分为-1
        perfect_logit = torch.where(mask == 1, torch.tensor(1.0), torch.tensor(-1.0))

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        logits = torch.div(
            self._cosine_simililarity(anchor_feature, contrast_feature),
            self.temperature if temperature is None else temperature)

        # tile mask

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # compute log_prob
        log_prob = cal_log_prob(logits, logits_mask)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        # loss
        if self.contrastive_method == 'cl':
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.view(anchor_count, batch_size).mean()
            # if temperature == 0.1:
            #     perfect_logit = torch.div(perfect_logit,self.temperature)
            #     perfect_log_prob = cal_log_prob(perfect_logit, logits_mask)
            #     perf = (mask * perfect_log_prob).sum(1) / (mask.sum(1) + 1e-6)
            #     perf_loss = - (self.temperature / self.base_temperature) * perf
            #     perf_loss = perf_loss.view(anchor_count, batch_size).mean()
            #     print("perfect:",perf_loss,"now", loss)
        elif self.contrastive_method == 'infoNCE':
            loss = loss
        else:
            raise ValueError('Unknown contrastive method: {}'.format(self.contrastive_method))

        return loss,logits,perfect_logit

def cal_log_prob(logits, logits_mask):
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    return log_prob

if __name__ == "__main__":
    criterion = ContrastiveLoss(contrastive_method='cl')
    a_input = torch.randn(4, 3, 4, 4)
    label = torch.tensor([1, 1, 0, 1])
    loss = criterion(a_input,
                     labels=label
                     )
