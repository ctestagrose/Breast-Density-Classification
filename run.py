from model_def import ModelDefinition
import train
import predict

cache = 1
debug = 0
batch_size = 8
num_class = 4
pretrained = True
dropout_ratio = 0.25
fc_nodes = 1024
patch_size = 32
img_size = 299
epochs = 50
model_path = "/content/drive/MyDrive/Breast_Density_Kaggle_Sample_Data/Models/Sample_Model"
if os.path.isdir(model_path) == False:
    os.makedirs("/content/drive/MyDrive/Breast_Density_Kaggle_Sample_Data/Models/Sample_Model")
# model_name = "InceptionV3"
model_name = 'ViT_Pretrained'
train(cache, batch_size, img_size, epochs, model_path, model_name, debug)
predict(model_name, model_path, num_class)
