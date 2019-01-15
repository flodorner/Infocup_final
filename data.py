import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import csv
from config import GTSRB_DIRECTORY, FACES_DIRECTORY, BB_LABELS_DIRECTORY


def read_labels(path):
    labels = pd.read_csv(path, header=None)
    return np.array(labels)[:, 1:].astype(float)


class GTSRB(Dataset):

    def __init__(self, bb_labels=True):
        print('Reading data')
        self.features, self.labels = self._get_training_data(GTSRB_DIRECTORY)
        if bb_labels:
            self.labels = read_labels(GTSRB_DIRECTORY + 'trafficsignbb.csv')
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

        if bb_labels:
            self.target_transforms = transforms.Lambda(lambda x: torch.from_numpy(x))
        else:
            self.target_transforms = transforms.Lambda(lambda x: torch.eye(43)[int(x)])  # 43 classes

        print('GTSRB loaded!')

    def __getitem__(self, index):
        X = self.features[index]
        X = self.transforms(X)
        X *= 255
        y = self.labels[index]
        y = self.target_transforms(y)
        return (X, y)

    def __len__(self):
        return len(self.features)

    @staticmethod
    def _read_annotations_file(prefix, gtFile):
        images = []
        labels = []
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
        next(gtReader)  # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0]))  # the 1st column is the filename
            labels += [row[7]]  # the 8th column is the label
        gtFile.close()
        return images, labels

    @staticmethod
    def _get_class(rootpath, c):
        prefix = rootpath + 'Final_Training/Images/' + format(c, '05d') + '/'  # subdirectory for class
        gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # annotations file
        images, labels = GTSRB._read_annotations_file(prefix, gtFile)
        return images, labels

    @staticmethod
    def _get_training_data(rootpath):
        train_images = []
        train_labels = []
        for c in range(0, 43):  # loop over all 43 classes
            images, labels = GTSRB._get_class(rootpath, c)
            train_images += images
            train_labels += labels
        return train_images, train_labels


class Faces(Dataset):

    def __init__(self, bb_labels=True):
        self.bb_labels = bb_labels
        images = []
        with open(BB_LABELS_DIRECTORY + 'facelabelsbb.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                images.append(plt.imread(FACES_DIRECTORY + str(row[0])[:-8] + '.ppm'))
        self.images = np.array(images)
        print('Sanity test: Image array has dimension {}.'.format(self.images.shape))
        self.transforms = transforms.ToTensor()
        if bb_labels:
            self.labels = read_labels(BB_LABELS_DIRECTORY + 'facelabelsbb.csv')
            self.target_transforms = transforms.Lambda(lambda x: torch.from_numpy(x))
        print('Faces loaded!')

    def __getitem__(self, index):
        X = self.images[index]
        X = self.transforms(X)
        if self.bb_labels:
            y = self.labels[index]
            y = self.target_transforms(y)
            return (X, y)
        return (X, 0)

    def __len__(self):
        return len(self.images)

