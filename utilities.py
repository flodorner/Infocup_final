from config import *
import numpy as np
from os import remove
import requests
from PIL import Image

classnamedict={'Zulässige Höchstgeschwindigkeit (20)':0,
               'Zulässige Höchstgeschwindigkeit (30)':1,
               'Zulässige Höchstgeschwindigkeit (50)':2,
               'Zulässige Höchstgeschwindigkeit (60)':3,
               'Zulässige Höchstgeschwindigkeit (70)':4,
               'Zulässige Höchstgeschwindigkeit (80)':5,
               'Ende der Geschwindigkeitsbegrenzung (80)':6,
               'Zulässige Höchstgeschwindigkeit (100)':7,
               'Zulässige Höchstgeschwindigkeit (120)':8,
               'Überholverbot für Kraftfahrzeuge aller Art':9,
               'Überholverbot für Kraftfahrzeuge mit einer zulässigen Gesamtmasse über 3,5t':10,
               'Einmalige Vorfahrt':11,
               'Vorfahrt':12,
               'Vorfahrt gewähren':13,
               'Stoppschild':14,
               'Verbot für Fahrzeuge aller Art':15,
               'Verbot für Kraftfahrzeuge mit einer zulässigen Gesamtmasse von 3,5t':16,
               'Verbot der Einfahrt':17,
               'Gefahrenstelle':18,
               'Kurve (links)':19,
               'Kurve (rechts)':20,
               'Doppelkurve (zunächst links)':21,
               'Unebene Fahrbahn':22,
               'Schleudergefahr bei Nässe oder Schmutz':23,
               'Fahrbahnverengung (rechts)':24, #Nicht geprüft!
               'Baustelle':25,
               'Lichtzeichenanlage':26, #Np
               'Fußgänger':27,
               'Kinder':28, #Np
               'Fahrradfahrer':29,
               'Schnee- oder Eisglätte':30, #Np
               'Wildwechsel':31,
               'Ende aller Streckenverbote':32, 
               'Ausschließlich rechts':33,
               'Ausschließlich links':34,
               'Ausschließlich geradeaus':35,
               'Ausschließlich geradeaus oder rechts':36, #np
               'Ausschließlich geradeaus oder links':37, #np
               'Rechts vorbei':38,
               'Links vorbei':39,
               'Kreisverkehr':40,
               'Ende des Überholverbotes für Kraftfahrzeuge aller Art':41, #np
               'Ende des Überholverbotes für Kraftfahrzeuge mit einer zulässigen Gesamtmasse über 3,5t':42}

def dict_parser(string):
    dictionary={}
    strings=string.split("}")[:-1]
    for substring in strings:
        classlabel=substring.split("\"")[3]
        confidence=float(substring.split(":")[-1])
        dictionary[classlabel]=confidence
    return dictionary

def class_labels_to_one_hot(output,n=43):
    output=dict_parser(output)
    labels=np.zeros(n)
    for key in output:
        labels[classnamedict[key]]=output[key]
    return labels

def query_to_text(image_path):
    files = {'image': (image_path, open(image_path, 'rb'), 'image/png')}
    data = {'key': KEY}
    r = requests.post(URL, files=files, data=data)
    if r.status_code != 200:
        raise Exception(str(r.status_code) + ' ' + r.text)
    return r.text
    
def query_to_labels(image_path):  
    return class_labels_to_one_hot(query_to_text(image_path))   

def url_to_im(imurl):
    im = Image.open(imurl)
    im = im.convert("RGB")
    im = np.asarray(im)
    im.setflags(write=1)
    return im

def save(im,save_url):
    save_im=Image.fromarray(im)
    save_im.save(save_url)
    return None

def save_and_query(im,save_url,delete=False):
    save(im,save_url)
    label=np.array(query_to_labels(save_url))
    if delete==True:
        remove("temp.png")
    return label





