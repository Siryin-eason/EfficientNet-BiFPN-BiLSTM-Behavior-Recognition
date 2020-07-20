#  EfficientNet-BiFPN-BiLSTM (Behavior recognition)

 An efficient and multi-scale feature fusion behavior recognition algorithm

Environment:

python>=3.6

pytorch>=1.2

The overall frame diagram is as follows:

![Image text]
(https://github.com/Siryin-eason/EfficientNet-BiFPN-BiLSTM-Behavior-Recognition/blob/master/img/main.png)

Datasets:

Following the format of UCF101 action recognition.

Run steps:

1. Modify the "dict_data" of readpkl.py to your own category to generate your own label.pkl for training.
2. Run train.py, remember to modify the data set address.
3. During operation, all loss npy files will be saved for visualization, and all models will be saved under weights.
4. Run train.py, remember to modify the test data set address and the name of your trained model.
In addition, Ranger and BiFPN modules included in this project could also be used for other computer vision tasks.
