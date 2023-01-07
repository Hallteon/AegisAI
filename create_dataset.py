import os
from pathlib import Path

images = sorted(os.listdir('dataset_images/'))

for i, img in enumerate(images):
    sfx = Path(f"dataset_images/{img}").suffix
    os.rename(f'dataset_images/{img}', f'images/{i}{sfx}')