import pandas as pd

data = [
    ['The', 'Business', 'Centre', '15', 'Stevenson', 'Lane'],
    ['6', 'Mossvale', 'Road'],
    ['Studio', '7', 'Tottenham', 'Court', 'Road']
]

class Address(object):
    def __init__(self, *address):
        if not address:
            self.address = None
            print('No address given')
        else:
            self.address = ' '.join(str(x) for x in address)



class ModelHyperparameters(object):
    def __init__(self, **hyperparams):
        if not hyperparams:
            self.hyperparams = None
        else:
            self.hyperparams = hyperparams



