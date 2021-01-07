CONFIG_DICT = {
    "emb_sz": 400,
    "n_hid": 1152,
    "n_layers": 3,
    "pad_token": 1,
    "qrnn": False,
    "bidir": False,
    "output_p": 0.4,
    "hidden_p": 0.3,
    "input_p": 0.4,
    "embed_p": 0.05,
    "weight_p": 0.5}

CLASSES = [0, 1]
# TO GET CLASSES from fastai learner > learner.data.classes

VOCAB_SIZE = 15000

# TO GET VOCAB_SIZE from fastai learner > len(learn.data.vocab.itos)

THRESHOLD_PREDICTION = 0.5