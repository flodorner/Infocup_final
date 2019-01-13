# coding=utf-8
from config import *

import numpy as np
from os import remove
import requests
from PIL import Image
from time import sleep


def dict_parser(string):
    dictionary = {}
    strings = string.split("}")[:-1]
    for substring in strings:
        classlabel = substring.split("\"")[3]
        confidence = float(substring.split(":")[-1])
        dictionary[classlabel] = confidence
    return dictionary


def class_labels_to_one_hot(output):
    output = dict_parser(output)
    labels = np.zeros(LABEL_AMOUNT)
    for key in output:
        labels[CLASSNAMEDICT[key]] = output[key]
    return labels


class ServerError(Exception):
    def __init__(self, message):
        super().__init__(message)


def query_to_text(image_path, retries=5):
    files = {'image': (image_path, open(image_path, 'rb'), 'image/png')}
    data = {'key': KEY}
    for i in range(retries):
        r = requests.post(URL, files=files, data=data)
        if r.status_code != 200:
            print("Query failed. Status code:")
            print(str(r.status_code) + ' ' + r.text)
            if i == retries - 1:
                raise ServerError("Too many retries. " + str(r.status_code) + ' ' + r.text)
            else:
                print("Trying again")
                sleep(1)
        else:
            break
    return r.text


def query_to_labels(image_path):  
    return class_labels_to_one_hot(query_to_text(image_path))   


def url_to_array(imurl):
    im = Image.open(imurl)
    # Avoid unintend behaviour for RGBA-Images
    if np.array(im).shape[2] == 4:
        new_image = Image.new("RGBA", im.size, "WHITE")
        new_image.paste(im, (0, 0), im)
        im = new_image
    im = im.convert("RGB")
    im = np.asarray(im)
    im.setflags(write=True)
    return im


def url_to_torch(imurl):
    im = url_to_array(imurl)
    im = im.astype(float)
    im = im.reshape(1, im.shape[2], im.shape[0], im.shape[1])
    return im


def save(im, save_url):
    save_im = Image.fromarray(im)
    save_im.save(save_url)
    return None


def save_and_query(im, save_url, delete=False):
    save(im, save_url)
    label = np.array(query_to_labels(save_url))
    if delete:
        try:
            remove("save_url")
        except FileNotFoundError:
            pass
    return label


def query_names(im_url):
    dictionary = dict_parser(query_to_text(im_url))
    for key in dictionary:
        if dictionary[key] != 0:
            prob = dictionary[key]
            dictionary[key] = (key + " (label " + str(CLASSNAMEDICT[key]) + "): " + str(prob))
    return dictionary


def torch_to_array(im):
    out_im = np.array(im)
    out_im = out_im.reshape(out_im.shape[1], out_im.shape[2], out_im.shape[0])
    out_im = np.maximum(np.minimum(out_im, np.zeros(out_im.shape) + 255), np.zeros(out_im.shape))
    out_im = out_im.astype(np.uint8)
    return out_im
