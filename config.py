import json
import os
import random


class Config(object):
    def __init__(self, config):
        self.config = config
        self.dataset = config.get("dataset")
        self.downstream = config.get("downstream")
        self.method = config.get("method")

        exp_id = config.get("exp_id", None)
        if exp_id is None:
            exp_id = config["exp_id"] = int(random.SystemRandom().random() * 100000)

        self._parse_default_config()
        self._parse_config_file()
        self._parse_external_config()

    def __getitem__(self, name):
        return self.config.get(name, None)

    def get(self, name, default=None):
        return self.config.get(name, default)

    def _parse_file_into_config(self, file_name):
        with open(file_name, 'r') as f:
            json_obj = json.load(f)
            for key in json_obj:
                self.config[key] = json_obj[key]

    def _parse_default_config(self):
        """
        parse default config from
            config/downstream/{downstream},
            config/method/{method},
            and dataset/{dataset}/config.json.
        """
        self._parse_file_into_config("config/downstream/{}.json".format(self.downstream))
        self._parse_file_into_config("config/method/{}.json".format(self.method))
        self._parse_file_into_config("dataset/{}/config.json".format(self.dataset))

    def _parse_config_file(self):
        """
        parse user config file.
        """
        config_file = self.config.get("config_file", None)
        if config_file is not None:
            if os.path.exists(config_file + ".json"):
                self._parse_file_into_config(config_file + ".json")
            else:
                raise FileNotFoundError(
                    'Config file {}.json is not found. Please ensure \
                    the config file is in the root dir and is a JSON \
                    file.'.format(config_file))

    def _parse_external_config(self):
        pass
