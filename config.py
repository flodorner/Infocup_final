URL = 'https://phinau.de/trasi'
KEY = 'ut6ohb7ZahV9tahjeikoo1eeFaev1aef'
STICKER_DIRECTORY = "Quickstick"
LABEL_AMOUNT = 43
IMAGE_SIZE = 64
CHANNELS = 3

# Write instructions and default parameters to readme!
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
