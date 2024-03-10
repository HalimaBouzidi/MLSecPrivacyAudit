import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

import numpy as np
from PIL import Image

def train(model, num_epochs, optimizer, criterion, train_loader, val_loader, device, path):
    
    if optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                          lr=0.001,
                          momentum=0.9,
                          weight_decay=1e4)
        
    elif optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    else:
        raise NotImplementedError


    model.to(device)

    early_stopping = EarlyStopping(patience=35, verbose=False, path=path+'/checkpoint.pt')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, classes in train_loader:
            images = images.to(device)
            classes = classes.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, classes)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        running_loss = 0.0
        
        model.eval()

        eval_len = 0

        for images, classes in val_loader:
            images = images.to(device)
            classes = classes.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, classes)

            running_loss += loss.item() * images.size(0)
            eval_len += images.size(0)

        val_loss = running_loss / eval_len

        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping at Epoch: {}".format(epoch))
            break

    return model


def test(model, test_loader, device, criterion=None):

    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0

    eval_len = 0

    with torch.no_grad():
        for images, classes in test_loader:
            images = images.to(device)
            classes = classes.to(device)

            eval_len += images.size(0)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            if criterion is not None:
                loss = criterion(outputs, classes).item()
            else:
                loss = 0

            running_loss += loss * images.size(0)
            running_corrects += torch.sum(preds == classes.data).item()

    eval_accuracy = running_corrects/eval_len
    eval_loss = running_loss/eval_len

    return eval_loss, eval_accuracy

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    