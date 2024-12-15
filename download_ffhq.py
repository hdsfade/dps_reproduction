import os
import random
from datasets import load_dataset

dataset = load_dataset("merkol/ffhq-256", split="train")

random_indices = random.sample(range(len(dataset)), 100)
print(dataset)
for idx in random_indices:
    image = dataset[idx]["image"]
    image.save(f"data/samples/{idx:05d}.png")