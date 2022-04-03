class Config(object):
    def __init__(self, config):
        self.config = config

    def get(self, name, default=None):
        return self.config.get(name, default)
