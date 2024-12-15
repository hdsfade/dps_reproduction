from functools import partial
import os
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.condition_inverser import ConditionInverserFactory
from guided_diffusion.measurements import OperatorFactory
from guided_diffusion.noise import NoiseFactory
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import BetaScheduleFactory, SamplerFactory
from guided_diffusion.posterior_calculator import PosteriorCalculatorFactory
from data.dataloader import get_dataset, get_dataloader
from guided_diffusion.helpers import clear_color, mask_generator
from guided_diffusion.logger import get_logger

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()
   

    logger = get_logger()

    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  

    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    measure_config = task_config['measurement']
    operator_factory = OperatorFactory()
    operator = operator_factory.create_operator(device=device, **measure_config['operator'])

    noise_factory = NoiseFactory()
    noiser = noise_factory.create_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    cond_config = task_config['conditioning']
    condition_inverser = ConditionInverserFactory()
    cond_method = condition_inverser.create_condition_inverser(cond_config['method'], operator, noiser, cond_config['params']['scale'])
    measurement_cond_fn = cond_method.condition_inverse
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    beta_schedule_factory = BetaScheduleFactory()
    posterior_calculator_factory = PosteriorCalculatorFactory()
    sampler_factory = SamplerFactory(beta_schedule_factory, posterior_calculator_factory)
    sampler = sampler_factory.create_sampler(**diffusion_config)
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
   
    out_path = os.path.join(args.save_dir, task_config['conditioning']['method'],measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    batch_size = args.batch_size
    loader = get_dataloader(dataset, batch_size=batch_size, num_workers=0, train=False)

    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )
        
    for i, ref_imgs in enumerate(loader):
        logger.info(f"Inference for image batch {i}, batch_size: {batch_size}")

        ref_imgs = ref_imgs.to(device)

        if measure_config['operator'] ['name'] == 'inpainting':
            mask = mask_gen(ref_imgs)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            cond_method.operator.set_mask(mask)
            measurement_cond_fn = partial(cond_method.condition_inverse)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            y = operator.forward(ref_imgs)
            y_n = noiser(y)

        else: 
            y = operator.forward(ref_imgs)
            y_n = noiser(y)

        x_start = torch.randn(ref_imgs.shape, device=device).requires_grad_()
        sample = sample_fn(x0=x_start, y=y_n, record_step=0, save_root=out_path)
        for j in range(batch_size):
            fname = str(i*batch_size+j).zfill(5) + '.png'
            plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n[j]))
            plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_imgs[j]))

            plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample[j]))

if __name__ == '__main__':
    main()
