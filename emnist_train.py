import torch
from emnist import extract_training_samples
import torch_cnn
import itertools
import matplotlib.pyplot as plt
import math

def train(num_epochs, cnn, train_loader, valid_loader, optimiser, loss_func, scheduler=None):
    '''Trains the CNN with given parameters'''
    valid_accuracy = []
    for epoch in range(num_epochs):
        cnn.train()
        for (images, labels) in train_loader:
            images, labels = images.cuda(), labels.cuda()
            output = cnn.forward(images)
            loss = loss_func(output, labels)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        
        if scheduler != None:
            scheduler.step()

        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        valid_accuracy.append(valid(cnn, valid_loader).item())
    return valid_accuracy

def valid(cnn, valid_loader):
    '''Calculates the accuracy of the CNN on the validation data'''
    cnn.eval()
    with torch.no_grad():
        correct = 0
        for images, labels in valid_loader:
            images, labels = images.cuda(), labels.cuda()
            test_output = cnn.forward(images)
            y = torch.max(test_output, 1)[1]
            correct += (y == labels).sum()
    accuracy = correct / 400 # Our validation data has 40,000 images
    print('Validation Data Accuracy: {0:.2f}'.format(accuracy))
    return accuracy

def train_plot_params(epochs, train_data, valid_data):
    '''Plots accuracy on validation data for all combinations of given parameters'''
    # Take subset of data to save time
    train_data = train_data[:50000]
    # Parameters
    fmaps1s = [40]
    fmaps2s = [160]
    denses = [200]
    droupouts = [0.4]
    batch_sizes = [5]

    # Train and save accuracies
    plots = []
    for params in list(itertools.product(fmaps1s, fmaps2s, denses, droupouts, batch_sizes)):
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=params[4], shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=10000, shuffle=False)
        cnn = torch_cnn.EMNISTCNN(*params[:4])
        cnn.cuda()

        loss_func = torch.nn.CrossEntropyLoss()
        optimiser = torch.optim.SGD(cnn.parameters(), lr = 0.02*math.sqrt(params[4]))
        scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.5)

        plots.append((train(epochs, cnn, train_loader, valid_loader, optimiser, loss_func, scheduler=scheduler), *params))
    
    # Plot accuracies
    for plot in plots:
        plt.plot(range(epochs), plot[0], label=str(plot[1:])[1:-1])
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('validation accuracy')
    plt.title('Validation accuracy of fmaps1, fmaps2, dense, dropout, batch_size')
    plt.savefig('validation_plot.png')

def train_save(epochs, train_data, valid_data):
    # Train the network and save it
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=5, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=10000, shuffle=False)

    cnn = torch_cnn.EMNISTCNN(40, 160, 200, 0.4)
    cnn.cuda()
    valid(cnn, valid_loader)

    loss_func = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(cnn.parameters(), lr = 0.045)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=15, gamma=0.5)

    train(epochs, cnn, train_loader, valid_loader, optimiser, loss_func, scheduler=scheduler)
    torch.save(cnn, 'torch_emnistcnn.pt')

if __name__ == '__main__':
    epochs = 60
    # Load EMNIST training dataset
    images, labels = extract_training_samples('digits')
    train_images = torch.tensor((images[:200000]/255-0.5).reshape(200000, 1, 28, 28))
    train_labels = labels[:200000]
    valid_images = torch.tensor((images[200000:]/255-0.5).reshape(40000, 1, 28, 28))
    valid_labels = labels[200000:]
    train_data = list(zip(train_images.float(), train_labels.astype('int64')))
    valid_data = list(zip(valid_images.float(), valid_labels.astype('int64')))

    train_save(epochs, train_data, valid_data)
    #train_plot_params(epochs, train_data, valid_data)