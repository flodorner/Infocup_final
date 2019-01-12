# coding=utf-8
from config import *

import numpy as np
from os import remove
import requests
from PIL import Image
from time import sleep

classnamedict = {'Zulässige Höchstgeschwindigkeit (20)': 0,
                 'Zulässige Höchstgeschwindigkeit (30)': 1,
                 'Zulässige Höchstgeschwindigkeit (50)': 2,
                 'Zulässige Höchstgeschwindigkeit (60)': 3,
                 'Zulässige Höchstgeschwindigkeit (70)': 4,
                 'Zulässige Höchstgeschwindigkeit (80)': 5,
                 'Ende der Geschwindigkeitsbegrenzung (80)': 6,
                 'Zulässige Höchstgeschwindigkeit (100)': 7,
                 'Zulässige Höchstgeschwindigkeit (120)': 8,
                 'Überholverbot für Kraftfahrzeuge aller Art': 9,
                 'Überholverbot für Kraftfahrzeuge mit einer zulässigen Gesamtmasse über 3,5t': 10,
                 'Einmalige Vorfahrt': 11,
                 'Vorfahrt': 12,
                 'Vorfahrt gewähren': 13,
                 'Stoppschild': 14,
                 'Verbot für Fahrzeuge aller Art': 15,
                 'Verbot für Kraftfahrzeuge mit einer zulässigen Gesamtmasse von 3,5t': 16,
                 'Verbot der Einfahrt': 17,
                 'Gefahrenstelle': 18,
                 'Kurve (links)': 19,
                 'Kurve (rechts)': 20,
                 'Doppelkurve (zunächst links)': 21,
                 'Unebene Fahrbahn': 22,
                 'Schleudergefahr bei Nässe oder Schmutz': 23,
                 'Fahrbahnverengung (rechts)': 24,
                 'Baustelle': 25,
                 'Lichtzeichenanlage': 26,
                 'Fußgänger': 27,
                 'Kinder': 28,
                 'Fahrradfahrer': 29,
                 'Schnee- oder Eisglätte': 30,
                 'Wildwechsel': 31,
                 'Ende aller Streckenverbote': 32,
                 'Ausschließlich rechts': 33,
                 'Ausschließlich links': 34,
                 'Ausschließlich geradeaus': 35,
                 'Ausschließlich geradeaus oder rechts': 36,
                 'Ausschließlich geradeaus oder links': 37,
                 'Rechts vorbei': 38,
                 'Links vorbei': 39,
                 'Kreisverkehr': 40,
                 'Ende des Überholverbotes für Kraftfahrzeuge aller Art': 41,
                 'Ende des Überholverbotes für Kraftfahrzeuge mit einer zulässigen Gesamtmasse über 3,5t': 42}

reverse_classnamedict = {v: k for k, v in classnamedict.items()}


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
        labels[classnamedict[key]] = output[key]
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
            dictionary[key] = (key + " (label " + str(classnamedict[key]) + "): " + str(prob))
    return dictionary


def torch_to_array(im):
    out_im = np.array(im)
    out_im = out_im.reshape(out_im.shape[1], out_im.shape[2], out_im.shape[0])
    out_im = np.maximum(np.minimum(out_im, np.zeros(out_im.shape) + 255), np.zeros(out_im.shape))
    out_im = out_im.astype(np.uint8)
    return out_im
