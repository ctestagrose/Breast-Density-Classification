import json
import random
import torch
import cv2
import torchvision.transforms
from monai.transforms import (
    LoadImaged,
    Resized,
    NormalizeIntensityd,
    AddChanneld,
    RandFlipd, RandRotated,
    RepeatChanneld,
    AsChannelFirstd,
    EnsureTyped,
    Compose)
from Crop_Image import CropImage, CropImaged
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from monai.data import CacheDataset, DataLoader, Dataset
from model_def import ModelDefinition

def train(cache, batch_size, img_size, epochs, model_path, model_name, debug):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open("./data_dict.json") as f:
        data = json.load(f)

    train_data = data["Train"]
    validation_data = data["Validation"]
    random.Random(42).shuffle(train_data)
    random.Random(42).shuffle(validation_data)

    if debug == 1:
        train_data = train_data[:100]
        validation_data = validation_data[:100]

    MONAI_train_transforms = Compose(
        [LoadImaged(keys=['image']),
         AddChanneld(keys=["image"]), RepeatChanneld(keys=["image"], repeats=3),
         Resized(spatial_size=(int(img_size), int(img_size)), keys=['image']),
         RandFlipd(keys=['image'], prob=0.5), RandRotated(range_x=2, keys=['image'], prob=0.5),
         NormalizeIntensityd(keys=['image']),
         EnsureTyped(keys=['image', 'label'])])

    MONAI_validation_transforms = Compose(
        [LoadImaged(keys=['image']),
         AddChanneld(keys=["image"]), RepeatChanneld(keys=["image"], repeats=3),
         Resized(spatial_size=(int(img_size), int(img_size)), keys=['image']),
         NormalizeIntensityd(keys=['image']),
         EnsureTyped(keys=['image', 'label'])])

    pytorch_Train = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        CropImage(),
        torchvision.transforms.Resize((int(img_size), int(img_size))),
        torchvision.transforms.ToTensor()]
    )

    pytorch_Validate = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        CropImage(),
        torchvision.transforms.Resize((int(img_size), int(img_size))),
        torchvision.transforms.ToTensor()]
    )

    # temp_transforms = Compose(
    #     [LoadImaged(keys=['image']), AddChanneld(keys=["image"]), RepeatChanneld(keys=["image"], repeats=3 )])

    # for item in train_data:
    #     img = Image.open(item["image"])
    #     print(img.size)
    #     transformed = temp_transforms(item)
    #     print(transformed["image"].shape)
    #     print("\n")


    if int(cache) == 1:
        train_ds = CacheDataset(
            data=train_data,
            transform=MONAI_train_transforms,
            cache_rate=1,
            num_workers=16
        )
        train_loader = DataLoader(train_ds, batch_size=int(batch_size), num_workers=8)
        validation_ds = CacheDataset(
            data=validation_data,
            transform=MONAI_validation_transforms,
            cache_rate=1,
            num_workers=16
        )
        validation_loader = DataLoader(validation_ds, batch_size=int(batch_size), num_workers=8)

    else:
        train_ds = dataloader(train_data, transforms=pytorch_Train)
        validation_ds = val_dataloader(validation_data, transforms=pytorch_Validate)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=int(batch_size), num_workers=8)
        validation_loader = torch.utils.data.DataLoader(validation_ds, batch_size=int(batch_size), num_workers=8)

    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    val_interval = 1
    # writer = SummaryWriter()

    mod = ModelDefinition(num_class, pretrained_flag=pretrained, dropout_ratio=dropout_ratio, fc_nodes=fc_nodes,
                          patch_size=patch_size, img_size=img_size)

    if model_name == 'InceptionV3':
        model = mod.inception_v3()
        #model = nn.DataParallel(model)
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), 0.001)
    elif model_name == 'ViT_Pretrained':
        model = mod.ViT_Pretrained()
        #model = nn.DataParallel(model)
        train_loader = DataLoader(train_ds, batch_size=int(batch_size), num_workers=8)
        validation_loader = DataLoader(validation_ds, batch_size=int(batch_size), num_workers=8)
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)

    model.to(device)

    for epoch in range(epochs):
        print("-" * 10)
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data['image'].to(device), batch_data['label'].type(torch.float).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            # writer.add_scaler("Train Loss ", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss: .4f}")

        if (epoch + 1) % int(val_interval) == 0:
            model.eval()
            num_correct = 0.0
            metric_count = 0
            val_epoch_loss = 0
            val_step = 0
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in validation_loader:
                    val_step += 1
                    val_images, val_labels = val_data['image'].to(device), val_data['label'].type(torch.float).to(device)
                    val_output = model(val_images)
                    value = torch.eq(val_output.argmax(dim=1), val_labels.argmax(dim=1))
                    val_loss = loss_function(val_output, val_labels)
                    val_epoch_loss += val_loss.item()
                    metric_count += len(value)
                    num_correct += value.sum().item()
                    val_epoch_len = len(validation_ds) // validation_loader.batch_size
                    # writer.add_scaler("Validation Loss ", val_loss.item(), val_epoch_len * epoch + val_step)
                metric = num_correct / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), model_path+"/best_model_vit.pth")
                    print('Saved new model')
                print("Current Epoch: {} current accuracy: {:.4f}"
                      " Best accuracy: {:.4f} at epoch {}".format(epoch + 1, metric, best_metric, best_metric_epoch))
                print("val_accuracy", metric, epoch + 1)
    print(f"training completed, best_metric: {best_metric: .4f}"
                  f" at epoch: {best_metric_epoch}")
        # writer.close()
