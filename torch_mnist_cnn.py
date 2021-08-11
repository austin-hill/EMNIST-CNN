import torch
from torch import nn
import mnist

if __name__ == '__main__':
    class MNISTCNN(nn.Module):
        def __init__(self):
            super(MNISTCNN, self).__init__()
            self.conv1 = nn.Sequential(         
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding='valid'),                              
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2),
            )
            self.conv2 = nn.Sequential(         
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding='valid'),                              
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2),
            )
            self.fcon1 = nn.Sequential(nn.Linear(1024, 1024), nn.LeakyReLU())
            self.fcon2 = nn.Sequential(nn.Linear(1024, 128), nn.LeakyReLU())
            self.fcon3 = nn.Linear(128, 10)
            self.dropout = nn.Dropout(p=0.5)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(self.fcon1(x))
            x = self.dropout(self.fcon2(x))
            x = self.fcon3(x)
            return x

    use_saved = True

    train_images = (mnist.train_images()/255 - 0.5).reshape(60000, 1, 28, 28)
    train_labels = mnist.train_labels()
    test_images = (mnist.test_images()/255 - 0.5).reshape(10000, 1, 28, 28)
    test_labels = mnist.test_labels()
    train_images, test_images = map(torch.tensor, (train_images, test_images))

    train_data = list(zip(train_images.float(), train_labels.astype('int64')))
    test_data = list(zip(test_images.float(), test_labels.astype('int64')))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=False, num_workers=1)
    
    if use_saved == True:
        cnn = torch.load('torch_mnistcnn_checkpoint.pt')
    else:
        cnn = MNISTCNN()
    cnn.cuda()

    loss_func = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(cnn.parameters(), lr = 0.005)

    def test():
        cnn.eval()
        with torch.no_grad():
            correct = 0
            for images, labels in test_loader:
                images, labels = images.cuda(), labels.cuda()
                test_output = cnn.forward(images)
                pred_y = torch.max(test_output, 1)[1]
                correct += (pred_y == labels).sum()
        accuracy = correct / 100
        print('Test Accuracy: {0:.2f}'.format(accuracy))
        return accuracy

    def train(num_epochs, cnn):
        best_res = 99.61
        for epoch in range(num_epochs):
            cnn.train()
            for (images, labels) in train_loader:
                images, labels = images.cuda(), labels.cuda()
                output = cnn.forward(images)            
                loss = loss_func(output, labels)
                 
                optimiser.zero_grad()           
                loss.backward()                
                optimiser.step()

            print ('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

            test_res = test()
            if test_res >= best_res:
                best_res = test_res
                torch.save(cnn, 'torch_mnistcnn.pt')
    test()
    #train(40, cnn)