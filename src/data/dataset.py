import torch
from torch.utils.data import Dataset as dataset 

class Dataset_class(dataset):

    def __init__(self, data):
        #initialize dataset
        super().__init__(data)
        self.data = data
    


    def get_data(self,index):
        #Datenpunkt holen
        item = self.data(index)
        
        #Hier code schreiben um trainingsdaten einzulesen funktioniert noch nicht
        image = item['path']

        #concepts und label in tensor umwandeln
        concepts = torch.tensor(item["concepts"], dtype=torch.float32)
        label = torch.tensor(item["label"], dtype=torch.float32)

        return image, concepts, label
    

    def set_data(self, new_data):
        self.data = new_data