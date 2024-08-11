# Cassava Leaf Disease Classification using Transfer Learning

Cultivated extensively in the tropical regions, Cassava is a major source of energy and contains a lot of essential vitamins and minerals. In Uganda, it is grown both as cash and food security crops. However, it is prone to viral and bacterial diseases which causes heavy crop losses and affects food security. So, identifying and detecting these diseases is very important for the livelihood of farmers. 

Using a dataset containing more than 21,000 annotated photos taken during a routine survey in Uganda, this work aims to categorise each cassava image into four disease categories or a fifth category that indicates a healthy leaf. Transfer learning was used to train EfficientNet-B0, a Convolutional Neural Network (CNN) model. Neural network models extract the required features automatically to classify an image into its respective class. Transfer learning is a machine learning technique in which a model created for one task is used to do another task that is related to it. Various image augmentation techniques were used to overcome imabalance in data. 

The developed model was deployed as an interactive web application using Streamlit and this proposed system also attained an overall accuracy score of 0.8887 and 0.8851 on the private and public Kaggle competition leaderboards respectively, indicating the efficiency of the approach.

The dataset used was obtained from Kaggle and can be found here: https://www.kaggle.com/c/cassava-leaf-disease-classification/

**Screenshots of the web app:**

![image](https://github.com/user-attachments/assets/7c2c6918-e343-41be-82ac-297c94bf07fd)

Home page of the web app

![image](https://github.com/user-attachments/assets/76f6d2cc-6b2c-497e-b70f-a300dfd81b68)

Prediction result
