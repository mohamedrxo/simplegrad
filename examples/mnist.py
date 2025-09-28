from simplegrad import Tensor, SGD ,MSELoss,Linear,DataLoader
import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd


# Downloading the MNIST Dataset from kaggle
path = kagglehub.dataset_download("oddrationale/mnist-in-csv")
print("Path to dataset files:", path)

train = pd.read_csv(f"{path}/mnist_train.csv")
test = pd.read_csv(f"{path}/mnist_test.csv")

# spliting images and labels
images = train.values[:,1:]
labels = train.values[:,0]

#passing the data to a dataloader
traing_data = DataLoader(images,labels)

# Defining the MNIST Model Class
class MNISTModel():
    def __init__(self,in_features,hidden_features, out_features):
        self.linear1 = Linear(in_features, hidden_features,bias=True)
        self.linear2 = Linear(hidden_features, out_features,bias=True)
    def __call__(self, x):
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        return  x

    def parameters(self):
        return self.linear1.parameters() + self.linear2.parameters()

# Creating the model
model = MNISTModel(784,100,10)

# Creating the MSELoss and SGD optimizer
criterion  = MSELoss()
optimizer = SGD(model.parameters(),lr=0.01)

# The Training Loop
for i in range(10000):
    # every batch contain 20 samples from the MNIST Dataset
    batch = traing_data(20)
    input_images = Tensor(batch['data']/ 255.0)
    labels = Tensor.one_hotencoding(batch['label'],  10)

    predictions = model(input_images)
    predictions = predictions.softmax()
    loss = criterion(predictions,labels)
    loss.backward()
    print("batch: ",i+1,'loss: ',f"{loss.item():.5f}")
    optimizer.step()
    optimizer.zero_grad()


# Testing The Model
batch = traing_data(6)
input_images = Tensor(batch['data'] / 255.0)
labels = Tensor.one_hotencoding(batch['label'], 10)

# Grid layout
n_images = batch['data'].shape[0]
cols = 3  # 3 images per row
rows = (n_images + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
axes = axes.flatten()

for i in range(n_images):
    test_image = batch['data'][i]
    resized_img = test_image.reshape((28, 28))
    label = batch['label'][i]

    # Model prediction
    prediction = model(Tensor(np.expand_dims(test_image, axis=0)))
    prediction = prediction.softmax() * 100
    prediction = prediction.data.squeeze()
    pred_class = np.argmax(prediction)
    pred_prob = np.max(prediction)

    axes[i].imshow(resized_img, cmap="gray")
    axes[i].set_title(f"True: {label}\nPred: {pred_class} ({pred_prob:.1f}%)")
    axes[i].axis("off")

# Hide any unused axes
for j in range(n_images, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()