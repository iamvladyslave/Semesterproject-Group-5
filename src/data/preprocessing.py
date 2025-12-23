import torch
from torchvision import transforms

class Preprocessing:
    def __init__(self, image_size=(128, 128)):
        #initialize preprocessing with image size
        self.image_size = image_size
        #trangsformation for preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image):
        #apply transformations to the image
        return self.transform(image)