import numpy as np
from tensorflow.keras.datasets import mnist

#loading mnist data
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

#joining train and test data
mnist_data=np.vstack([trainData,testData])
mnist_label=np.hstack([trainLabels,testLabels])
print(mnist_data.shape,mnist_label.shape)

az_label=[]
az_data=[]

datasetPath='C:/Users/saura/Downloads/Compressed/A_Z Handwritten Data.csv/A_Z Handwritten Data.csv'

# loop over the rows of the A-Z handwritten digit dataset
for row in open(datasetPath):
    # parse the label and image from the row
    row = row.split(",")
    label = int(row[0])
    image = np.array([int(x) for x in row[1:]], dtype="uint8")
    image = image.reshape((28, 28))
    az_data.append(image)
    az_label.append(label)
az_data = np.array(az_data, dtype="float32")
az_label = np.array(az_label, dtype="int")
print(az_data.shape,az_label.shape)



# the MNIST dataset occupies the labels 0-9, so let's add 10 to every
# A-Z label to ensure the A-Z characters are not incorrectly labeled
# as digits
az_label += 10

data=np.vstack([az_data,mnist_data])
labels=np.hstack([az_label,mnist_label])


#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data,labels, test_size = 0.2)
print(Y_test)

from tensorflow.keras.utils import to_categorical
train_yOHE = to_categorical(Y_train,dtype='int')
print("New shape of train labels: ", train_yOHE.shape)

test_yOHE = to_categorical(Y_test, dtype='int')
print("New shape of test labels: ", test_yOHE.shape)

#make value btw 0 and 1 for easier calculation(preprocessing)

X_train=X_train/255.0
X_test=X_test/255.0
print(X_train.shape,X_test.shape)

X_train = X_train.reshape(353960, 28, 28,1)
X_test = X_test.reshape(88491, 28, 28,1)

from keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam

#sequential means passing data from one side of nn to other side of nn
model = Sequential()
#CNN
# input -> conv -> maxpool -> conv -> maxpool ......->flattened vector->
#.                        hidden layer -> hidden layer -> softmax layer
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Flatten())

model.add(Dense(64,activation ="relu"))
model.add(Dense(128,activation ="relu"))

model.add(Dense(36,activation ="softmax"))


model.summary()

model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, train_yOHE, epochs=3,batch_size=2, validation_data = (X_test,test_yOHE))

score = model.evaluate(X_test, test_yOHE, verbose=0)

print("Test Loss: %.2f%%" % (score[0]*100))
print("Accuracy: %.2f%%" % (score[1]*100))

model.save('smodel.h5')