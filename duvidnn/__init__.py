from importlib.metadata import version

from duvida.config import config
config.set_backend('torch', precision='float')

app_name = "duvidnn"
__author__ = "Eachan Johnson"
__version__ = version(app_name)
