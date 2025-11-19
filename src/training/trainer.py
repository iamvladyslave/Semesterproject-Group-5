import torch
import torch.nn as nn

class Trainer:
    def __init__(self, train_loader, val_loader, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_val_loss = 0.0
        self.best_model= None
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def train_epoch(self, dataloader: torch.utils.data.DataLoader, train = True):
        #setze Trainingsmodus oder Evaluationsmodus
        if train: 
            self.model.train()
        else:
            self.model.eval()
        #total initialisieren
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        #dataloader durchlaufen
        for images, labels in dataloader:
            #Daten verschieben
            images, labels = images.to(self.device), labels.to(self.device)
            #gradient berechnen
            with torch.set_grad_enabled(train):
                #compute logits
                logits = self.model(images)
                #compute loss
                loss = self.loss_fn(logits, labels)
                #backpropagation and optimalization
                if train:
                    self.optimizer.zero_grad()
                   
                    loss.backward()
                    self.optimizer.step()

                #sigmoid activation
                predicted = torch.sigmoid(logits > 0.5).float()
                #update total
                total_loss += loss.item()
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        #compute avg und accuracy 
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples

        return avg_loss, accuracy
    
    def training(self,epochs):
        #run throuhjh epochs
        for epoch in range(epochs):
            
            #training 
            train_loss = self.train_epoch(self.train_loader, train=True)
            train_acc = self.train_epoch(self.train_loader, train=True)
            #validation
            val_loss = self.train_epoch(self.train_loader, train=False)
            val_acc = self.train_epoch(self.val_loader, train=False)

            #history update
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            #print progress
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            #save best model
            if val_loss < self.best_val_loss:
                #update best loss
                self.best_val_loss = val_loss
                #safe model parameters
                self.best_model = self.model.state_dict()
                print("new best model")

        return self.history
       
        

    