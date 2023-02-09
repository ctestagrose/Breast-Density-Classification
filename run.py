import train
import predict
import os


if __name__ == "__main__":
    cache = 1
    debug = 0
    batch_size = 16
    num_class = 4
    pretrained = True
    dropout_ratio = 0.25
    fc_nodes = 1024
    patch_size = 32
    img_size = 299
    epochs = 100
    model_path = "./Models/Sample_Model"
    data_dictionary = "./data_dictionaries/data_dict.json"
    if os.path.isdir("./Models") == False:
        os.makedirs("./Models")
    if os.path.isdir(model_path) == False:
        os.makedirs("./Models/Sample_Model")
    # model_name = "InceptionV3"
    model_name = 'ViT_Pretrained'
    train.train(num_class, pretrained, cache, data_dictionary, batch_size, img_size, epochs, model_path, model_name, debug)
    predict.predict(model_name, model_path, num_class, data_dictionary, img_size)

