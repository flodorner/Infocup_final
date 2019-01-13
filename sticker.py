import numpy as np
from glob import glob
from os import remove
from time import sleep
from random import randint

from config import *
from utilities import url_to_array, save, save_and_query


def stick_trans(image, sticker):
    
    result = np.zeros(image.shape, dtype=np.uint8)
    mask = sticker == 255
    mask2 = sticker != 255
    result[mask] = 255
    result[mask2] = image[mask2]
    
    return result


def stick(image, sticker):
  
    result = np.zeros(image.shape, dtype=np.uint8)
    mask = np.mean(sticker, 2) > 0
    mask2 = np.mean(sticker, 2) == 0
    mask = np.stack((mask, mask, mask), -1)
    mask2 = np.stack((mask2, mask2, mask2), -1)
    result[mask] = sticker[mask]
    result[mask2] = image[mask2]
   
    return result


def load_random_sticker(label):
    imlist = glob(STICKER_DIRECTORY + "/" + str(label) + "/*")
    if len(imlist) == 0:
        raise Exception("no stickers with this label available")
    try:
        imlist.remove("/" + str(label) + "/desktop.ini")
    except ValueError:
        pass
    sticker_url = imlist[randint(0, len(imlist)-1)]
    return sticker_url


class StickerGenerator:

    def __init__(self, pixelsize=3, fringe=17, start=np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)):

        self.imagesize = IMAGE_SIZE
        self.pixelsize = pixelsize
        self.fringe = fringe
        self.start = start

        basic_im = np.copy(self.start)
        self.basic_prob = np.array(save_and_query(basic_im, "temp.png"))

        self.queries = 1
        self.num_rows = (self.imagesize - 2 * self.fringe) / self.pixelsize
        
        if self.num_rows.is_integer():
            self.num_rows = int(self.num_rows)
            self.probarray = np.zeros((self.num_rows, self.num_rows, 3, LABEL_AMOUNT))

        else:
            print("Error: Image size - 2 * fringe should be dividable by pixelsize. Generation of stickers"
                  "won't be possible without reinitalization!")
            
        try:
            remove("temp.png")
        except FileNotFoundError:
            pass

    def _generate_pixels(self):
        for i in range(self.num_rows):
            for j in range(self.num_rows):
                for k in range(3):
                    basic_im = np.copy(self.start)
                    basic_im[self.fringe + i * self.pixelsize:self.fringe + (i + 1) * self.pixelsize,
                             self.fringe + j * self.pixelsize:self.fringe + (j + 1) * self.pixelsize, k] = 255
                    prob = np.array(save_and_query(basic_im, "temp.png"))
                    self.probarray[i, j, k] = prob - self.basic_prob
                    self.queries += 1
                    sleep(1)
        try:
            remove("temp.png")
        except FileNotFoundError:
            pass
        return None

    def _make_sticker(self, label, title="", pixel_threshold=0.01, save_threshold=0.9):

        basic_im = np.copy(self.start)
        for i in range(self.num_rows):
            for j in range(self.num_rows):
                for k in range(3):
                    if self.probarray[i, j, k, label] > pixel_threshold:
                        basic_im[self.fringe + i * self.pixelsize:self.fringe + (i + 1) * self.pixelsize,
                                 self.fringe + j * self.pixelsize:self.fringe + (j + 1) * self.pixelsize, k] = 255
        if np.max(basic_im) > 0:                
            prob = np.array(save_and_query(basic_im, "temp.png"))
            self.queries += 1 
            sleep(1)
            print("Confidence for Sticker on label " + str(label) + ": " + str(prob[label]))
        else: 
            prob = np.zeros(LABEL_AMOUNT)
        if prob[label] > save_threshold:
            save_url = STICKER_DIRECTORY + "/" + str(label) + "/" + title +\
                       str(pixel_threshold) + str(prob[label]) + ".png"
            save(basic_im, save_url)
            print("Sticker saved under " + save_url)
        try:
            remove("temp.png")
        except FileNotFoundError:
            pass
        
        return basic_im

    def sticker_batch(self, title="", pixel_threshold=0.01, save_threshold=0.9):
        self._generate_pixels()
        for i in range(LABEL_AMOUNT):
            self._make_sticker(i, title=title, pixel_threshold=pixel_threshold, save_threshold=save_threshold)
        return None

    @staticmethod
    def sticker_attack(image_url, save_url, sticker_url=None, mode="full", label=None):
        if sticker_url is None:
            if label is None:
                label = 5
            sticker_url = load_random_sticker(label)

        elif label is None:
            label = int(sticker_url.split("/")[-2])

        sticker = url_to_array(sticker_url)
        image = url_to_array(image_url)

        if mode == "full":
            output = stick(image, sticker)
        elif mode == "transparent":
            output = stick_trans(image, sticker)
        else:
            raise Exception("mode is supposed to be full or transparent")

        conf = save_and_query(output, save_url)[label]

        return conf
