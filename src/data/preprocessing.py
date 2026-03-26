import torch
from torchvision import transforms

class Preprocessing:
    '''
    this class transforms images into a predefined size,
    converts them to tensors and normalizes them (based on imagenet statistics)
    '''
    def __init__(self, image_size=(128, 128)):
        '''
        initializes image preprocessing
        
        Parameters
        -----------
        image_size : int tuple
            Target size (height, width) to images will be resized
            
        '''
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image):
        '''
        apply transformation to an image

        Parameters
        ----------
        image : PIL.Image.Image
            Input image on which preprocessing is applied

        Returns
        --------
        torch.Tensor
            image is returned as a tensor

        Examples
        --------
        >>> from PIL import Image
        >>> img = Image.open("example.jpg")
        >>> img_pre = Preprocessing()
        >>> img_tensor = img_pre.preprocess_image(img)
        '''
        #apply transformations to the image
        return self.transform(image)