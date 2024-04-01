
dataset_map = {
    'tree': {
        'url': 'https://klacansky.com/open-scivis-datasets/bonsai/bonsai.idx',
        'x': 256,
        'y': 256,
        'z': 256
    },
    'carp': {
        'url': 'https://klacansky.com/open-scivis-datasets/carp/carp.idx',
        'x': 256,
        'y': 256,
        'z': 512
    },
    'teapot': {
        'url': 'https://klacansky.com/open-scivis-datasets/boston_teapot/boston_teapot.idx',
        'x': 256,
        'y': 256,
        'z': 100
    },
    'lobster': {
        'url': 'https://klacansky.com/open-scivis-datasets/lobster/lobster.idx',
        'x': 301,
        'y': 324,
        'z': 56
    }
}


class VolumeDatasetLoader:
    def __init__(self, name):
        self.name = name
        if name not in dataset_map.keys():
            raise Exception("Dataset not defined.")
        self.dataset = dataset_map[name]

    def get_url(self):
        return self.dataset['url']

    def get_xyz(self):
        return self.dataset['x'], self.dataset['y'], self.dataset['z']
