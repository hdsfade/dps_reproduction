import torch_fidelity
import lpips
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse


def cal_fid(method, task, label_path, recon_path):
    fid = torch_fidelity.calculate_metrics(input1=label_path, input2=recon_path,fid=True)
    print(f"{method}-{task} fid: {fid}")

def cal_lpips(method, task, label_path, recon_path):
    lpips_model = lpips.LPIPS(net='alex').cuda()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    lpips_score_total = 0
    count = 0
    for filename in sorted(os.listdir(label_path)):
        label_image_path = os.path.join(label_path, filename)
        recon_image_path = os.path.join(recon_path, filename)

        if os.path.isfile(label_image_path) and os.path.isfile(recon_image_path):
            label_image = transform(Image.open(label_image_path).convert('RGB')).unsqueeze(0).cuda()
            recon_image = transform(Image.open(recon_image_path).convert('RGB')).unsqueeze(0).cuda()

            lpips_score_total += lpips_model(label_image, recon_image)
            count += 1

    lpips_score = lpips_score_total / count
    print(f'{method}-{task} lpips: {lpips_score.item()}')
   


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--metric', type=str, default='fid')
    args = parser.parse_args()
    parent_path = os.path.join(args.root_dir, args.method, args.task)
    label_path = os.path.join(parent_path, 'label')
    recon_path = os.path.join(parent_path, 'recon')

    if args.metric == 'fid':
        cal_fid(args.method, args.task, label_path, recon_path)
    elif args.metric == 'lpips':
        cal_lpips(args.method, args.task, label_path, recon_path)

if __name__ == '__main__':
    main()