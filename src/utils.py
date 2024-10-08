from torch import nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


class Pipeline:
    def __init__(self):

        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        print(self.device)

        transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])  # Normalize for grayscale images

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        images = torch.load("/home/skushwaha/DLGroup2/src/tensor_images.pt")
        labels = torch.load("/home/skushwaha/DLGroup2/src/labels.pt")
        dataset = torch.utils.data.TensorDataset(images, labels)
        combined_dataset = torch.utils.data.ConcatDataset([trainset, dataset])
        self.trainloader = torch.utils.data.DataLoader(combined_dataset, batch_size=32, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.lossFunc = nn.CrossEntropyLoss()

    def train_step(self, model, optimizer):
        model.train()
        epochloss = 0
        for batchcount, (images, labels) in enumerate(self.trainloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # labels = torch.eye(len(classes))[labels].to(device)
            # pdb.set_trace()
            
            optimizer.zero_grad()

            y = model(images)

            loss = self.lossFunc(y, labels)     
            loss.backward()

            optimizer.step()
            
            epochloss += loss.item()

        return epochloss

    def val_step(self, model):
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        model.eval()
        with torch.no_grad():
            for batchcount, (images, labels) in enumerate(self.testloader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                y = model(images)

                _, predicted = torch.max(y, 1)

                for label_index in range(labels.size(dim=0)):     
                    label_num = labels[label_index].item()
                    predicted_num = predicted[label_index].item()

                    y_true.append(classes[label_num])
                    y_pred.append(classes[predicted_num])

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            cm = confusion_matrix(y_true, y_pred, labels=classes)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
            disp.plot(cmap=plt.cm.Blues)
            plt.title('Confusion Matrix', fontsize=15, pad=20)
            plt.xlabel('Prediction', fontsize=11)
            plt.ylabel('Actual', fontsize=11)
            #Customizations
            plt.gca().xaxis.set_label_position('top')
            plt.gca().xaxis.tick_top()
            plt.gca().figure.subplots_adjust(bottom=0.2)
            plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)

            plt.show()

        return correct*100/total


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

LOSS_FN = nn.CrossEntropyLoss()
