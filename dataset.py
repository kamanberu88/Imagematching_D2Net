from PIL import ImageOps
from PIL import Image
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt

class D2NetDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        #print(img_name)
        image = Image.open(img_name)

        transformed_image = image.copy()

        # Convert PIL Image to numpy arrayzen
        transformed_image = np.array(transformed_image)

        # Randomly choose a type of noise to add
        noise_type = random.choice(['brightness', 'contrast', 'blur', 'noise', 'radiation'])
        #noise_type = random.choice([ 'contrast', 'blur', 'noise', 'radiation'])
        
        if noise_type == 'brightness':
            n_brightness = random.uniform(-0.2, 0.2)
            transformed_image = transformed_image + n_brightness * 255
        

        elif noise_type == 'contrast':
            n_contrast = random.uniform(-1, 1)
            transformed_image = ((transformed_image / 255.0 - 0.5) * 10 ** n_contrast + 0.5) * 255

        elif noise_type == 'blur':
            n_blur = random.uniform(0, 5)
            transformed_image = cv2.GaussianBlur(transformed_image, (5, 5), n_blur)
        
        elif noise_type == 'noise':
            n_noise = random.uniform(0, 0.01)
            row, col, ch = transformed_image.shape
            gauss = np.random.normal(0, n_noise, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            ransformed_image = transformed_image + transformed_image * gauss
    
        elif noise_type == 'radiation':
            n_radiation = random.uniform(0, 0.1)
            num_salt = np.ceil(n_radiation * transformed_image.size)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in transformed_image.shape]
            transformed_image[coords] = 255
            num_pepper = np.ceil(n_radiation * transformed_image.size)
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in transformed_image.shape]
            transformed_image[coords] = 0

        # Clip values to be in valid range [0, 255]
        transformed_image = np.clip(transformed_image, 0, 255)

        # Convert numpy array back to PIL Image
        transformed_image = Image.fromarray(np.uint8(transformed_image))
        
        


        sample = {'image1': image, 'image2': transformed_image}





        if self.transform:
            sample['image1'] = self.transform(image)
            #print(sample['image1'])
            sample['image2'] = self.transform(transformed_image)
            #print(sample['image2'])


        return sample



