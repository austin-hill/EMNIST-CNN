import torch
from emnist import extract_test_samples

def test(cnn):
    '''Calculates the accuracy of the CNN on the test data'''
    cnn.eval()
    with torch.no_grad():
        correct = 0
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            test_output = cnn.forward(images)
            pred_y = torch.max(test_output, 1)[1]
            correct += (pred_y == labels).sum()
    accuracy = correct / 400 # Our test data has 40,000 images
    print('Test Data Accuracy: {0:.2f}'.format(accuracy))
    return accuracy

if __name__ == '__main__':
    # Load EMNIST training dataset
    test_images, test_labels = extract_test_samples('digits')
    test_images = torch.tensor((test_images/255-0.5).reshape(40000, 1, 28, 28))
    test_data = list(zip(test_images.float(), test_labels.astype('int64')))

    # Load and test CNN
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=False)
    cnn = torch.load('torch_emnistcnn_checkpoint.pt')
    cnn.cuda()
    test(cnn)