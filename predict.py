import json
import random
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
import cv2
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from monai.config import print_config
from monai.data import Dataset, DataLoader, CacheDataset, decollate_batch
from monai.config import print_gpu_info
from model_def import ModelDefinition
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import (
    Activations,
    LoadImaged,
    Resized,
    RandFlipd, RandRotated,
    ToPILd,
    AsChannelFirstd, NormalizeIntensityd,
    EnsureTyped,
    AddChanneld,
    RepeatChanneld,
    EnsureType,
    Compose)
import torch
import torchvision
import progressbar
from Crop_Image import CropImage, CropImaged


class dataloader(Dataset):
    def __init__(self, dict, transforms):
        self.dict = dict
        self.transforms = transforms

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, index):
        image = cv2.imread(self.dict[index]['image'])
        image = self.transforms(image)
        label = self.dict[index]['label']
        label = torch.FloatTensor(label)
        return image, label

def returnCAM(model_path, model_name):
    print_config()
    print_gpu_info()

    test_transforms = Compose(
        [LoadImaged(keys=['image']),
         AddChanneld(keys=["image"]), RepeatChanneld(keys=["image"], repeats=3),
         Resized(spatial_size=(int(img_size), int(img_size)), keys=['image']),
         NormalizeIntensityd(keys=['image']),
         EnsureTyped(keys=['image', 'label'])])

    with open("/content/drive/MyDrive/Breast_Density_Kaggle_Sample_Data/data_dict_u.json") as f:
        data = json.load(f)

    test_list = data['Test']

    random.Random(42).shuffle(test_list)
    test_list = test_list[:25]

    counter = 0
    for i in test_list:
        orig = test_transforms(i)
        orig = orig['image']
        cv2.imwrite(model_path + '/Saliency_Maps/orig' + str(counter) + '.png', orig)
        counter += 1

    labels = []
    for item in test_list:
        labels.append(item['label'])

    test_ds = CacheDataset(
        data=test_list,
        transform=test_transforms,
        cache_rate=1,
        num_workers=16
    )

    model_filename = model_path + '/best_model.pth'

    num_classes = int(sys.argv[3])

    # place list of classes here in the format of 'Class Identifier':[], 'Class Identifier':[]"
    class_list = {}

    count = 0
    for _class in class_list:
        label_list = [0] * num_classes
        label_list[count] = 1
        class_list[_class] = label_list
        count += 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == "InceptionV3":
        model_def = mod.inception_v3()
        model = model_def.get_model()
        #model = nn.DataParallel(model)
    if model_name == 'ViT':
        model_def = mod.ViT_Pretrained()
        model = model_def.get_model_ViT()
        #model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(torch.load(model_filename))
    model.eval()

    params = list(model.parameters())
    data = params[0].cpu().data.numpy()
    weight = np.squeeze(data)
    print('weight.shape', weight.shape)

    test_loader = DataLoader(test_ds, num_workers=0)

    counter = 0

    for _testImage in test_loader:
        image = _testImage['image'].to(device)
        gt = _testImage['label']
        image.requires_grad_()
        output = model(image)
        output = model(image)

        output_idx = output.argmax()
        output_max = output[0, output_idx]
        output_max.backward()
        saliency, _ = torch.max(image.grad.data.abs(), dim=1)
        saliency = saliency.reshape(299, 299)

        image = image.reshape(-1, 299, 299)
        image = image.cpu().detach().numpy().transpose(1, 2, 0)

        index = test_list[counter]
        transformed_image = test_transforms(index)
        transformed_image = transformed_image['image']
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(transformed_image.astype(np.uint8))
        ax[0].axis('off')
        ax[1].imshow(saliency.cpu(), cmap='hot')
        ax[1].axis('off')
        from textwrap import wrap
        temp = test_list[counter]
        plt.imshow(transformed_image.astype(np.uint8))
        saliency = saliency.cpu()
        print(saliency)
        plt.imshow(cv2.rotate(np.float32(saliency), cv2.ROTATE_90_CLOCKWISE), cmap='hot', alpha=0.7)
        plt.tight_layout()
        _fn = model_path+'/Saliency_Maps/saliency_map_' + str(counter) + '.png'
        plt.savefig(_fn)
        plt.clf()
        counter += 1

def generate_ROC(y_test, y_score, Y_PRED_NP, num_classes, GT_LIST, model_name, model_path):
    y_test = np.array([t.ravel() for t in y_test])
    y_score = np.array([t.ravel() for t in Y_PRED_NP])

    cycol = cycle('rgbymc')
    cylin = cycle('-:.')
    lw = 2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    labels = ['A', 'B', 'C', 'D']

    for i in range(num_classes):
        if i % 6 == 0:
            lin = next(cylin)
        fpr[i], tpr[i], _ = roc_curve(np.array(y_test[:, i]), np.array(Y_PRED_NP[:, i]))
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=next(cycol), ls=lin, lw=lw, label='AUC: {0:0.2f} - {1}'
                                                                         ''.format(roc_auc[i], labels[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves of Multi-Class Classification')
    plt.legend(loc="lower right")
    _fn = model_path + '/ROC_curve_' + model_name + '.png'
    plt.savefig(_fn)
    print('Area under the ROC curve is, ', roc_auc)
    plt.clf()

def generate_confusion_matrix(y_true, y_predicted, class_list, model_name):
    cf_matrix = confusion_matrix(y_true, y_predicted)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in class_list], columns=[i for i in class_list])
    df_cm.to_csv(model_path + '/confusion_matrix_' + model_name + '.csv')
    num_correct = df_cm.iloc[0, 0] + df_cm.iloc[1, 1] + df_cm.iloc[2, 2] + df_cm.iloc[3, 3]
    return num_correct

def predict(model_name, model_path, num_class):
    print_config()
    print_gpu_info()

    test_transforms = Compose(
        [LoadImaged(keys=['image']),
         AddChanneld(keys=["image"]), RepeatChanneld(keys=["image"], repeats=3),
         Resized(spatial_size=(int(img_size), int(img_size)), keys=['image']),
         NormalizeIntensityd(keys=['image']),
         EnsureTyped(keys=['image', 'label'])])

    pytorch_Test = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        CropImage(),
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor()]
    )

    with open("/content/drive/MyDrive/Breast_Density_Kaggle_Sample_Data/data_dict_u.json") as f:
        data = json.load(f)

    test_list = data['Test']

    random.Random(42).shuffle(test_list)

    labels = []
    for item in test_list:
        labels.append(item['label'])

    pytorch_transforms = "No"

    if pytorch_transforms == 'No':
        test_ds = CacheDataset(
            data=test_list,
            transform=test_transforms,
            cache_rate=1,
            num_workers=16
        )

    model_filename = model_path + '/best_model_vit.pth'

    num_classes = int(num_class)

    #place list of classes here in the format of 'Class Identifier':[], 'Class Identifier':[]"
    class_list = {}

    count = 0
    for _class in class_list:
        label_list = [0] * num_classes
        label_list[count] = 1
        class_list[_class] = label_list
        count += 1

    out_filename = model_path+'/Predictions.xlsx'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    if model_name == "InceptionV3":
        model_def = mod.inception_v3()
        model = model_def.get_model()
        #model = nn.DataParallel(model)
    if model_name == 'ViT_Pretrained':
        model_def = mod.ViT_Pretrained()
        model = model_def.get_model_ViT()
        #model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(torch.load(model_filename))
    model.eval()

    y_pred_trans = Compose([EnsureType(), Activations(sigmoid=True)])

    count = 0
    Image_FN_LIST = []
    Y_PRED_NP = np.zeros([len(test_list), num_classes])
    y_predicted = []
    y_true = []
    y_test = []
    y_score = []
    GT_LIST = []

    pb = progressbar.ProgressBar(max_value=len(test_list))

    if pytorch_transforms == 'Yes':
        test_ds = dataloader(test_list, transforms=pytorch_Test)
        test_loader = torch.utils.data.DataLoader(test_ds, num_workers=4)
    else:
        test_loader = DataLoader(test_ds, num_workers=0)

    #labels should be in the format of ['Class', 'Class', 'Class'....]
    labels = []
    #valies should be in the format of ['0', '1'...]
    values = []

    if pytorch_transforms == 'Yes':
        for image in test_list:
            Image_FN_LIST.append(image['image'])
        with torch.no_grad():
            for _testImage in test_loader:
                image = torch.as_tensor(_testImage[0]).to(device)
                y_test.append(np.array(_testImage[1]))
                gt = _testImage[1].cpu().detach().numpy()
                GT_LIST.append(labels[int(np.argmax(gt[0]))])
                y_true.append(values[int(np.argmax(gt[0]))])
                y = model(image)
                y_pred = y_pred_trans(y).to('cpu')
                y_predicted.extend(values[torch.argmax(y_pred)])
                Y_PRED_NP[count, :] = y_pred
                pb.update(count)
                count += 1
    else:
        for image in test_list:
            Image_FN_LIST.append(image['image'])
        with torch.no_grad():
            for _testImage in test_loader:
                image = torch.as_tensor(_testImage['image']).to(device)
                y_test.append(np.array(_testImage['label']))
                gt = _testImage['label'].cpu().detach().numpy()
                GT_LIST.append(labels[int(np.argmax(gt[0]))])
                y_true.append(values[int(np.argmax(gt[0]))])
                y = model(image)
                y_pred = y_pred_trans(y).to('cpu')
                y_predicted.extend(values[torch.argmax(y_pred)])
                Y_PRED_NP[count, :] = y_pred
                pb.update(count)
                count += 1

    y_test = np.array([t.ravel() for t in y_test])
    y_score = np.array([t.ravel() for t in Y_PRED_NP])

    generate_ROC(y_test, y_score, Y_PRED_NP, num_classes, GT_LIST, model_name, model_path)

    num_correct = generate_confusion_matrix(y_true, y_predicted, class_list, model_name)

    df = pd.DataFrame(Y_PRED_NP, columns=class_list)
    df.insert(0, 'GT', GT_LIST)
    df.insert(1, 'File', Image_FN_LIST)
    df.to_excel(out_filename)

    kappa = metrics.cohen_kappa_score(y_true, y_predicted)
    print(kappa)

    with open(model_path+ '/classification_report_' + model_name + '.txt', 'w') as f:
        f.write(classification_report(y_true, y_predicted))
