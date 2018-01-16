import torch
from torch.autograd import Variable
import numpy as np
from torch import nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from wideresnet import WideResNet
import foolbox
from foolbox.models import PyTorchModel
import os
from PIL import Image

import types
import copy

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt





model = torch.load("./model_20.pt")

WideResNetMean = np.array([125.3, 123.0, 113.9], dtype=np.float32)
WideResNetStd = np.array([63.0, 62.1, 66.7], dtype=np.float32)


class CopyNormalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return copy_normalize(tensor, self.mean, self.std)


def copy_normalize(tensor, mean, std):
    t_list = []
    for t, m, s in zip(tensor.permute(1, 0, 2, 3), mean, std):
        t_list.append(t.sub(m).div(s))
    return torch.stack(t_list).permute(1, 0, 2, 3)


normalize = CopyNormalize(
    mean=[x / 255.0 for x in WideResNetMean.tolist()],
    std=[x / 255.0 for x in WideResNetStd.tolist()])


def denormalize(tensor, mean, std):
    t_list = []
    for t, m, s in zip(tensor.permute(1, 0, 2, 3), mean, std):
        t_list.append((t * s).add(m))
    return torch.stack(t_list).permute(1, 0, 2, 3)



def patch_model(model, prefuncs):
    # prefuncs is a list of preprocessing functions

    # seems copy.copy is already enough for this purpose
    new_model = copy.deepcopy(model)
    new_model.forward_orig = new_model.forward

    def forward(self, x):
        out = x
        for func in prefuncs:
            out = func(out)
        return self.forward_orig(out)

    new_model.forward = types.MethodType(forward, new_model)
    return new_model

kwargs = {'num_workers': 1, 'pin_memory': True}

val_set = datasets.ImageFolder(root="../classes_data_with_black/val", transform=transforms.ToTensor())

val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=1, shuffle=True, **kwargs)


model = patch_model(model, [normalize])

class DataTransform(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        super(DataTransform, self).__init__()
        self.dataset = dataset
        self.transform = transform
    def __getitem__(self, index):
        input = self.transform(self.dataset[index][0])
        target = self.dataset[index][1]
        return input, target
    def __len__(self):
        return len(self.dataset)


from foolbox.attacks.base import Attack
class OneStepGradSignAttack(Attack):
   """
   (inherited from foolbox)
   One-step fast gradient sign method
   Does not do anything if the model does not have a gradient.
   """
   def _apply(self, a, epsilon=0.1):
       if not a.has_gradient():
           return

       image = a.original_image
       min_, max_ = a.bounds()

       gradient = a.gradient()
       gradient_sign = np.sign(gradient) * (max_ - min_)

       perturbed = image + gradient_sign * epsilon
       perturbed = np.clip(perturbed, min_, max_)

       a.predictions(perturbed)

total = 0
not_find = 0
for i, (input, target) in enumerate(val_loader):
    
    input = input.numpy()[0]
    print(np.max(input))
    print(np.min(input))
    print(input.shape)
    target = target.numpy()[0]
    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10)
    criterion = foolbox.criteria.Misclassification()
    attack = OneStepGradSignAttack(fmodel, criterion)

    adversarial = attack(input, target, epsilon=0.1, unpack=False)

    if adversarial.distance.value != 0:
        if adversarial.image is None:
            not_find += 1
        else:
            input_img = denormalize(torch.from_numpy(input[np.newaxis, :, :, :]), [x / 255.0 for x in WideResNetMean.tolist()], [x / 255.0 for x in WideResNetStd.tolist()])
            adv_image = denormalize(torch.from_numpy(adversarial.image[np.newaxis, :, :, :]), [x / 255.0 for x in WideResNetMean.tolist()], [x / 255.0 for x in WideResNetStd.tolist()])
            new_ori = DataTransform(input, transforms.ToPILImage())
            new_adv = DataTransform(adversarial.image, transforms.ToPILImage())
            input_img = new_ori.dataset
            adv_image = new_adv.dataset
            input_img = input_img * 255
            adv_image = adv_image * 255
            input_img = input_img.astype('uint8').transpose(1, 2, 0)
            adv_image = adv_image.astype('uint8').transpose(1, 2, 0)
            im = Image.fromarray(input_img)
            im_adv = Image.fromarray(adv_image)
            im_diff = Image.fromarray(input_img - adv_image)
            im.save('../ori.png', "PNG")
            im_adv.save('../adv.png', "PNG")
            im_diff.save('../diff.png', "PNG")
        total += 1
    if total >= 1:
        break
print(not_find * 1.0 / total)