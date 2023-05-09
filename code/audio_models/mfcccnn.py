'''
- The model consists of **3 x "convolutional + pooling layer"**, then a fully connected layer. 
- The input is the 20 mfccs. Each convolutional layer have different number of convolutional filters. 
- Max pooling is used and **dropout** of 20% on the 2nd and 3rd convolutional layer is applied to reduce over-fitting.
- Then, **flatten** the layer before passing it to fully connected dense layer.
- It has **1 dense layer** with 128 neurons and 50% dropout is applied as well for better generalisation and reduce overfit on the training data.
- The final **output layer have 5 neurons**, which is the **5 categories** that would like the model to classify the audio. 
- The activiation for all the layers is 'relu' and 'softmax' at the final layer. **Softmax** makes the output sum up to 1 so the output can be interpreted as probabilities. 
The model will then make its prediction based on which option has the highest probability.

Reference:
[1] Lequn Chen et al. 
In-Situ Crack and Keyhole Pore Detection in Laser Directed Energy Deposition Through Acoustic Signal and Deep Learning. 
Available at SSRN: https://ssrn.com/abstract=4308023 or http://dx.doi.org/10.2139/ssrn.4308023 


'''

from torch import nn
from torchsummary import summary
import torch.nn.functional as F



class MFCCCNN(nn.Module):
    def __init__(self):
        super(MFCCCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.dropout1 = nn.Dropout(p=0.2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.dropout2 = nn.Dropout(p=0.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=28* 128, out_features=128) #72*128 for 100ms; 28*128 for 40ms
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        logits = self.fc2(x)
        predictions = self.softmax(logits)
        return predictions



if __name__ == "__main__":
    cnn = MFCCCNN()
    print (cnn)
    summary(cnn.cuda(), (1, 32, 18)) # mel-spectrogram - (1,32,18) 128*8 for 100 ms; 


