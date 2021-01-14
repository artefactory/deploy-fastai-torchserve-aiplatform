import json

CONFIG_DICT = {
    'emb_sz': 400,
    'n_hid': 1152,
    'n_layers': 3,
    'pad_token': 1,
    'bidir': False,
    'output_p': 0.4,
    'hidden_p': 0.3,
    'input_p': 0.4,
    'embed_p': 0.05,
    'weight_p': 0.5}

CLASSES = [0, 1]

with open("vocab.json", "r") as f:
    VOCAB = json.load(f)
VOCAB_SIZE = len(VOCAB)
