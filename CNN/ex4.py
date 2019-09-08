import torch
from torch import nn

from gcommand_loader import GCommandLoader
from cnnModel import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(train_loader, validation_loader):
    global learning_rate, epochs, model
    
    loss_criterion = nn.CrossEntropyLoss()
    # go through epochs
    for e in range(epochs):
        model.train()
        train_loss = 0
        total = 0
        
        for i, (spectograms, labels) in enumerate(train_loader):
            spectograms = spectograms.to(device)
            labels = labels.to(device)
            optim.zero_grad()
            outputs = model.cuda()(spectograms)
            loss = loss_criterion(outputs, labels)
            train_loss += loss.item()
            label_size = labels.size(0)
            total += label_size
            loss.backward()
            optim.step()

        validation_loss, accuracy = validate(model, validation_loader)
        print('Epoch [{}/{}], Train loss: {:.4f}, Validation loss: {:.4f}, Accuracy : {:.2f}%'
              .format(e + 1, epochs, (train_loss/total), validation_loss, accuracy))

##    return model


def validate(model, validate_loader):
    model.eval()
    loss_criterion = nn.CrossEntropyLoss()
    num_loss = 0
    total = 0
    correct = 0
    for i, (spectograms, labels) in enumerate(validate_loader):
        spectograms = spectograms.to(device)
        labels = labels.to(device)
        outputs = model.cuda()(spectograms)
        loss = loss_criterion(outputs, labels)
        num_loss += loss.item()

        label_size = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).cpu().sum().item()
        total += label_size

    return (num_loss/total), ((correct /total)*100)


def test(model, test_loader):
    model.eval()
    with open ("test_y", "w") as file:
        for i, (spectograms, labels) in enumerate(test_loader):
            spectograms = spectograms
            outputs = model(spectograms)
            root = test_loader.dataset.spects[i][0]
            splited_root = root.split('/')
            length = len(splited_root)
            _, predicted = torch.max(outputs.data, 1)
            file.write(splited_root[length-1]+", " + str(predicted.item()) + ".\n")


if __name__ == '__main__':
    epochs = 1
    num_classes = 30
    batch_size = 100
    learning_rate = 0.01

    model = CnnModel()
    
    print("Adam Optim:")
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train_set = GCommandLoader('./data/train')
    # validation_set = GCommandLoader('./data/valid')
    test_set = GCommandLoader('./data/test')

    # train_loader = torch.utils.data.DataLoader(
    #     train_set, batch_size=batch_size, shuffle=True,
    #     num_workers=20, pin_memory=True, sampler=None)
    #
    # validation_loader = torch.utils.data.DataLoader(
    #     validation_set, batch_size=1, shuffle=False,
    #     num_workers=20, pin_memory=True, sampler=None)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False,
        num_workers=20, pin_memory=True, sampler=None)

    # train(train_loader, validation_loader)
    test(model, test_loader)
