import numpy as np
from PIL import Image
from monai.config import KeysCollection
from typing import Optional, Any, Mapping, Hashable, Dict
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms import (
    MapTransform,
    Transform,
)

class CropImage(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self) -> None:
        roi_center: Union[Sequence[int], NdarrayOrTensor, None] = None,
        roi_size: Union[Sequence[int], NdarrayOrTensor, None] = None

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        # center_of_mass = mahotas.center_of_mass(img)
        check = str(img)
        if "PIL" in check:
            img = np.array(img)
            orig = img
            width = img.shape[0]

            sumh = np.sum(img, axis=0)
            white = np.where(sumh > 8 * width)
            ymin, ymax = np.min(white[0]), np.max(white[0])
            img = img[0:img.shape[0], ymin:ymax]

            height = img.shape[1]

            sumh = np.sum(img, axis=1)
            white = np.where(sumh > 12 * height)
            xmin, xmax = np.min(white[0]), np.max(white[0])
            img = img[xmin:xmax, 0:img.shape[1]]

            if img.shape[0] == 1 or img.shape[1] == 1:
                print("rejected crop")
                return orig

            img = Image.fromarray(img)
        else:
            orig = img
            width = img.shape[0]

            sumh = np.sum(img, axis=0)
            white = np.where(sumh > 8 * width)
            ymin, ymax = np.min(white[0]), np.max(white[0])
            img = img[0:img.shape[0], ymin:ymax]

            height = img.shape[1]

            sumh = np.sum(img, axis=1)
            white = np.where(sumh > 12 * height)
            xmin, xmax = np.min(white[0]), np.max(white[0])
            img = img[xmin:xmax, 0:img.shape[1]]

            if img.shape[0] == 1 or img.shape[1] == 1:
                print("rejected crop")
                return orig

            img = cv2.merge((img, img, img))

            # print(img.shape)

        return img

class CropImaged(MapTransform):
    backend = CropImage.backend

    def __init__(self, keys: KeysCollection) -> None:
        super().__init__(keys)
        self.cropper = CropImage()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            # self.push_transform(d[key])
            d[key] = self.cropper(d[key])
        return d
