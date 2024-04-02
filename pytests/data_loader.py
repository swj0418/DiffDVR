
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
        'z': 178
    },
    'lobster': {
        'url': 'https://klacansky.com/open-scivis-datasets/lobster/lobster.idx',
        'x': 301,
        'y': 324,
        'z': 56
    },
    'frog': {
        'url': 'https://klacansky.com/open-scivis-datasets/frog/frog.idx',
        'x': 256,
        'y': 256,
        'z': 44
    },
    'snake': {
        'url': 'https://klacansky.com/open-scivis-datasets/kingsnake/kingsnake.idx',
        'x': 1024,
        'y': 1024,
        'z': 795
    },
    'engine': {
        'url': 'https://klacansky.com/open-scivis-datasets/engine/engine.idx',
        'x': 256,
        'y': 256,
        'z': 128
    },
    'bag': {
        'url': 'https://klacansky.com/open-scivis-datasets/backpack/backpack.idx',
        'x': 512,
        'y': 512,
        'z': 373
    },
    'christmas': {
        'url': 'https://klacansky.com/open-scivis-datasets/christmas_tree/christmas_tree.idx',
        'x': 512,
        'y': 499,
        'z': 512
    },
    'chameleon': {
        'url': 'https://klacansky.com/open-scivis-datasets/chameleon/chameleon.idx',
        'x': 1024,
        'y': 1024,
        'z': 1000
    },
    'visible_male': {
        'url': 'https://klacansky.com/open-scivis-datasets/vis_male/vis_male.idx',
        'x': 128,
        'y': 256,
        'z': 256
    },
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
