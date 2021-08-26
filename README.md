# EMNIST-Convolutional-Neural-Net
A convolutional neural network implemented in PyTorch that achieved a 99.71% classification accurracy on the EMNIST dataset of digits after 60 epochs of training, without using an ensemble of networks. I included a checkpoint at the 99.71% accurracy.
# Requirements
```
pip install emnist matplotlib torch==1.9.0+cu111
```
# Sources
I chose to use two convolutional layers and two fully connected layers based on results from https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist. The rest of the parameters I chose myself using the plots generated by the train_plot_params function. Credit for the EMNIST dataset goes to: Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373.