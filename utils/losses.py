import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HingeLoss(nn.Module):
    '''
    Geometric GAN
    https://arxiv.org/abs/1705.02894'''
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:  # for D
            loss_real = F.relu(1 - pred_real).mean()
            loss_fake = F.relu(1 + pred_fake).mean()
            loss = loss_real + loss_fake
            return loss
        else:  # for G
            loss = -pred_real.mean()
            return loss


def gradient_normalize(f, x, **kwargs):
    """
    GN:
                             f
            f_hat = --------------------
                    || grad_f || + | f |
    《Gradient Normalization for Generative Adversarial Networks》https://arxiv.org/abs/2109.02235
    """
    grad = torch.autograd.grad(
        f, [x], torch.ones_like(f), create_graph=True, retain_graph=True)[0]
    grad_norm = torch.norm(torch.flatten(grad, start_dim=1), p=2, dim=1)
    grad_norm = grad_norm.view(-1, *[1 for _ in range(len(f.shape) - 1)])
    f_hat = (f / (grad_norm + torch.abs(f)))
    return f_hat






