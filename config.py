URL = 'https://phinau.de/trasi'
KEY = 'ut6ohb7ZahV9tahjeikoo1eeFaev1aef'
STICKER_DIRECTORY = "Quickstick"
FACES_DIRECTORY = "Faces/"
GTSRB_DIRECTORY = 'GTSRB/'
LABEL_AMOUNT = 43
IMAGE_SIZE = 64
WHITEBOX_DIRECTORY = "Models/ResNet.pt"
GAN_DIRECTORY = "Models/Generator.pt"
BB_LABELS_DIRECTORY = 'BBLabels/'

FGSM_SPECS = {
    "mode": "l_inf",
    "bound": 10,
    "magnitude": 1,
    "max_fgsm_iterations": 25,
    "target_threshold": 0.99,

    "fgsm_restart": "last",
    "restart_max_amount": 10,
    "restart_accuracy_bound": 0.0001,

    "retrain_mode": "last",
    "retrain_lr": 0.00001,
    "retrain_max_gradient_steps": 10,
    "retrain_threshold": 0.0001,
    "always_save": True,
    "print": True
}

GAN_SPECS = {
    "alpha": 10,
    "beta": 0.1,
    "epsilon": 0.05,
    "batch_size": 256,
    "cuda": True,
    "modelsaving_directory": '',
    "use_faces_dataset": True
}

CLASSNAMEDICT = {'Zulässige Höchstgeschwindigkeit (20)': 0,
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

REVERSE_CLASSNAMEDICT = {v: k for k, v in CLASSNAMEDICT.items()}
