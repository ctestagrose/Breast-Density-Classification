# Breast-Density-Classification
Code for the development of the Breast Density Classification Algorithm used in my Thesis. This code utilizes publicly available sample data.

Sample data can be found and downloaded at https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset

Before Running Code:
  1. Download the sample data set from the above kaggle webpage. 
  2. Place the jpeg folder found in the downloaded files into the project folder
  3. Install the required packages found in requirements.txt

Training Classifier:
  1. Make adjustments to the hyperparameters found in the run.py
  2. After adjusting the hyperparameters and file paths to your liking, run the following command in terminal
  ```
  python run.py
  ```
  
 Evaluating Classifier
  1. This code is set up to automatically begin evaluation after training for the specified number of epochs.      To adjust this, simply comment out the following line in run.py
  ```
  train.train(num_class, pretrained, cache, data_dictionary, batch_size, img_size, epochs, model_path, model_name, debug)
  ```
  2. Upon completion of the code, the results will be saved to the path specified in run.py
