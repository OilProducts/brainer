import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


class LocallyConnectedLayer(nn.Module):
    """
    A locally connected layer is a layer where each neuron is connected to only a small region of the input volume.
    The extent of this connectivity is a hyperparameter called the receptive field of the neuron (equivalently this is the filter size).
    The weights of the locally connected layer are not shared across spatial locations.

    Attributes:
        input_channels (int): Number of input channels.
        output_channels (int): Number of output channels.
        output_size (tuple): The height and width of the output.
        kernel_size (int): The size of the kernel.
        stride (int): The stride of the convolution.
        weights (torch.nn.Parameter): The weights for the locally connected layer.
        bias (torch.nn.Parameter): The bias for the locally connected layer.
    """

    def __init__(self, input_channels, output_channels, output_size, kernel_size, stride,
                 device='cpu'):
        """
        Initialize the LocallyConnectedLayer.

        Args:
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels.
            output_size (tuple): The height and width of the output.
            kernel_size (int): The size of the kernel.
            stride (int): The stride of the convolution.
        """
        super(LocallyConnectedLayer, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.output_size = output_size
        self.device = device

        # Define the weights for the locally connected layer
        self.weights = nn.Parameter(torch.randn(
            output_size[0], output_size[1],
            input_channels, kernel_size, kernel_size,
            output_channels
        )).to(self.device)
        self.bias = nn.Parameter(torch.randn(output_size[0], output_size[1], output_channels))

    def forward(self, x):
        """
        Forward pass of the locally connected layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Initialize output tensor
        output = torch.zeros(x.size(0), self.output_channels, *self.output_size).to(x.device)

        # Apply the locally connected operation
        for i in range(self.output_size[0]):
            for j in range(self.output_size[1]):
                for k in range(self.output_channels):
                    start_i = i * self.stride
                    start_j = j * self.stride
                    end_i = start_i + self.kernel_size
                    end_j = start_j + self.kernel_size

                    # Element-wise multiplication and sum
                    output[:, k, i, j] = (x[:, :, start_i:end_i, start_j:end_j] *
                                          self.weights[i, j, :, :, :, k]).sum(dim=(1, 2, 3)) + \
                                         self.bias[i, j, k]

        return output


# Define the neural network
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Locally Connected Layer
        # MNIST images are 28x28 pixels and grayscale (1 channel)
        self.local_conn = LocallyConnectedLayer(
            input_channels=1,
            output_channels=10,  # Arbitrary choice, can be tuned
            output_size=(24, 24),  # Output size considering 5x5 kernels and stride of 1
            kernel_size=5,
            stride=1,
            device=self.device
        )

        # Additional layers
        self.fc1 = nn.Linear(10 * 24 * 24, 128, device=self.device)  # Fully connected layer
        self.fc2 = nn.Linear(128, 10,
                             device=self.device)  # Output layer (10 classes for digits 0-9)

    def forward(self, x):
        x = F.relu(self.local_conn(x))  # Apply locally connected layer with ReLU activation
        x = x.view(-1, 10 * 24 * 24)  # Flatten the output for the fully connected layer
        x = F.relu(self.fc1(x))  # First fully connected layer with ReLU activation
        x = self.fc2(x)  # Output layer
        return x


# Initialize the network
model = MNISTNet()

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders
batch_size = 128
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

# Now you can proceed to train the model using train_loader and evaluate it using test_loader
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Common choice for classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Adam optimizer with a learning rate of
# 0.001


# Training loop
def train(model, train_loader, criterion, optimizer, epochs):
    model.train()  # Set the model to training mode
    print(model.device)
    for epoch in range(epochs):
        progress_bar = tqdm(iter(train_loader), total=len(train_loader),
                            unit_scale=train_loader.batch_size)
        total_loss = 0
        for data, target in progress_bar:
            # Move data and target to the same device as model
            data, target = data.to(model.device), target.to(model.device)

            # Forward pass
            output = model(data)

            # Compute loss
            loss = criterion(output, target)

            # Backward pass and optimize
            optimizer.zero_grad()  # Clear existing gradients
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Adjust the model weights

            total_loss += loss.item()

            # Print progress
            progress_bar.set_description(f'Epoch: {epoch + 1}/{epochs}, Loss: {loss.item()}')

        # Print average loss per epoch
        average_loss = total_loss / len(train_loader.dataset)

        # Now implement the test after each epoch
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():  # Do not calculate gradients to speed up computation
            test_progress_bar = tqdm(iter(test_loader), total=len(test_loader),
                                     unit_scale=test_loader.batch_size)
            for data, target in test_progress_bar:
                data, target = data.to(model.device), target.to(model.device)
                output = model(data)
                pred = output.argmax(dim=1,
                                     keepdim=True)  # Get the index of the class with the highest probability
                correct += pred.eq(
                    target.view_as(pred)).sum().item()  # Get total number of correct samples
                total += data.size(0)
                test_progress_bar.set_description(f'Test accuracy: {correct / total}')

    # Print test accuracy
    # print(f"Test accuracy: {correct / len(test_loader.dataset)}")


# Training the model
num_epochs = 10  # You can adjust the number of epochs
train(model, train_loader, criterion, optimizer, num_epochs)
