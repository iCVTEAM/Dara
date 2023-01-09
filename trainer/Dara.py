import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import ImageEnhance

sys.path.append(os.path.abspath('.'))
from train import Trainer
from utils import accuracy



transformtypedict = dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast,
                         Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)


class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (randtensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out

        
class DaraTrainer(Trainer):
    def __init__(self):
        super(DaraTrainer, self).__init__()

    def get_transformers(self, config):
        transformers = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop((160, 160)),
                ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]),
            'val': transforms.Compose([
                transforms.Resize(int(160 * 1.15)),
                transforms.CenterCrop(160),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        }
        return transformers

    def get_criterion(self, config):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, config):
        return torch.optim.SGD(self.model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay, nesterov=config.nesterov)

    def get_scheduler(self, config):
        return torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=config.decay_epoch, gamma=config.gamma)

    def batch_training(self, data):
        images, labels = self.to_device(data['img']), self.to_device(data['label'])

        # forward
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # record accuracy and loss
        acc = accuracy(outputs, labels, 1)
        self.average_meters['acc'].update(acc, images.size(0))
        self.average_meters['loss'].update(loss.item(), images.size(0))

    def batch_validate(self, data):
        images, labels = self.to_device(data['img']), self.to_device(data['label'])

        logits = self.model(images)
        acc = accuracy(logits, labels, 1)
        self.average_meters['acc'].update(acc, images.size(0))


if __name__ == '__main__':
    trainer = DaraTrainer()
    # print(trainer.model)
    trainer.train()
