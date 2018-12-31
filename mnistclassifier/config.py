import yaml


class Config():
    """
    Config representing a project
    """

    keys = []
    current = -1

    default = {
        'dataset': 'dataset.yml',
        'model': 'vgg16',
        'modelfile': 'model.save',
        'epochs': 2,
        'batch_size': 1,
        'learn_rate': 1.0
    }

    def __init__(self, configs={}):
        """
        args:
            config: Filename to a yaml file to load or dict config.
        """
        if isinstance(configs, dict):
            self.configs = configs
        else:
            with open(configs) as file:
                self.configs = yaml.load(file)
                self.filename = configs
        self.keys = list(self.configs.keys())
        for _ in self:
            self.fill_missing(self.get())

    def fill_missing(self, config):
        for key in self.default.keys():
            if config.get(key) is None:
                config[key] = self.default[key]

    def __iter__(self):
        self.current = -1
        return self

    def __next__(self):
        if self.current + 1 >= len(self):
            raise StopIteration
        else:
            self.current += 1
            return self.keys[self.current]

    def __getitem__(self, index):
        return self.configs[self.keys[index]]

    def __len__(self):
        return len(self.keys)

    def save(self, path):
        """
        save current config to given path as yaml file
        """
        with open(path, 'w') as outfile:
            yaml.dump({self.keys[self.current]: self[self.current]},
                      outfile, default_flow_style=False)

    def get(self, key=None):
        if key is None:
            key = self.keys[self.current]
        return self[self.keys.index(key)]

    def current_name(self):
        return self.keys[self.current]
