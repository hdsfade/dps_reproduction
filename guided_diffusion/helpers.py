import numpy as np
import torch.nn as nn
import scipy
from motionblur.motionblur import Kernel
import torch

def extract(a, t, x_shape):
    """
    Extract values from a tensor `a` at indices `t` and reshape to `x_shape`.
    
    Args:
        a (torch.Tensor): Tensor to extract from.
        t (torch.Tensor): Indices to extract.
        x_shape (tuple): Shape to reshape the output.
    
    Returns:
        torch.Tensor: Extracted and reshaped tensor.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def space_timesteps(timesteps, space_counts):
    if isinstance(space_counts, str):
        if space_counts.startswith('ddim'):
            desired_count = int(space_counts[len('ddim'):])
            pos_space = (timesteps - 1) // (desired_count-1)
            return set(range(0, timesteps, pos_space))
        space_counts = [int(x) for x in space_counts.split(',')]
    elif isinstance(space_counts, int):
        space_counts = [space_counts]
    per_steps = timesteps // len(space_counts)
    extra = timesteps % len(space_counts)
    start_idx = 0
    all_indices = []
    for i, count in enumerate(space_counts):
        size = per_steps + (1 if i < extra else 0)
        if size < count or count < 1:
            raise ValueError(f"Invalid step space in {space_counts} - per step size {size}")
        elif count == 1:
            space = 1
        else:
            space = (size - 1) // (count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += space
        all_indices.append(taken_steps)
        start_idx += size
    return all_indices

def clear_color(x):
    x = x.detach().cpu().squeeze().numpy()
    return normalize_np(np.transpose(x, (1, 2, 0)))


def normalize_np(img):
    img -= np.min(img)
    img /= np.max(img)
    return img

class Blurkernel(nn.Module):
    def __init__(self, blur_type='gaussian', kernel_size=31, std=3.0, device=None):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size//2),
            nn.Conv2d(3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3)
        )

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2,self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)
        elif self.blur_type == "motion":
            k = Kernel(size=(self.kernel_size, self.kernel_size), intensity=self.std).kernelMatrix
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k

class mask_generator:
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None,
                 image_size=256, margin=(16, 16)):
        """
        (mask_len_range): given in (min, max) tuple.
        Specifies the range of box size in each dimension
        (mask_prob_range): for the case of random masking,
        specify the probability of individual pixels being masked
        """
        assert mask_type in ['box', 'random', 'both', 'extreme']
        self.mask_type = mask_type
        self.mask_len_range = mask_len_range
        self.mask_prob_range = mask_prob_range
        self.image_size = image_size
        self.margin = margin

    def _retrieve_box(self, img):
        l, h = self.mask_len_range
        l, h = int(l), int(h)
        mask_h = np.random.randint(l, h)
        mask_w = np.random.randint(l, h)
        mask, t, tl, w, wh = random_sq_bbox(img,
                              mask_shape=(mask_h, mask_w),
                              image_size=self.image_size,
                              margin=self.margin)
        return mask, t, tl, w, wh

    def _retrieve_random(self, img):
        total = self.image_size ** 2
        # random pixel sampling
        l, h = self.mask_prob_range
        prob = np.random.uniform(l, h)
        mask_vec = torch.ones([1, self.image_size * self.image_size])
        samples = np.random.choice(self.image_size * self.image_size, int(total * prob), replace=False)
        mask_vec[:, samples] = 0
        mask_b = mask_vec.view(1, self.image_size, self.image_size)
        mask_b = mask_b.repeat(3, 1, 1)
        mask = torch.ones_like(img, device=img.device)
        mask[:, ...] = mask_b
        return mask

    def __call__(self, img):
        if self.mask_type == 'random':
            mask = self._retrieve_random(img)
            return mask
        elif self.mask_type == 'box':
            mask, t, th, w, wl = self._retrieve_box(img)
            return mask
        elif self.mask_type == 'extreme':
            mask, t, th, w, wl = self._retrieve_box(img)
            mask = 1. - mask
            return mask