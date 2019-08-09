# Image Classification
Here I have created a classificaion model with custom architecture. The model is trained using fashion-mnist dataset. Which is already in the keras package . I have used keras to create and train the model

## Network Architecture
input_layer -> CNN(32,(5,5)) -> Max-Pooling(2,2) -> Dropout(0.3) -> CNN(64,(3,3)) -> Max-Pooling(2,2) -> Dropout(0.5)-> CNN(64,(3,3)) -> Max-Pooling(2,2) -> Dropout(0.6) -> Dense(128) -> Dense(10)

CNN(x,(a,b) : x number of filter and (a,b) is the kernal size

The last Dense layer will be the output layer. All CNN's and Dense layer use relu activation exept the last Dense layer it uses softmax

## To train the model 

`python trainer.py --num_epochs=100`

if number of epochs is not given the default is set to 10. Also the train function has a callback to stop the training if the model overfits by monitoring the loss.

After training the model will be saved in the path same folder as custom_model.h5


