import os
from abc import ABC
from abc import abstractmethod


def download_github_code(path):
    filename = path.rsplit('/')[-1]
    os.system('shred -u {}'.format(filename))
    os.system(
        'wget -q https://raw.githubusercontent.com/hse-cs-ami/coursera-intro-dl/main/{} -O {}'.format(path, filename))


def download_github_release(path):
    filename = path.rsplit('/')[-1]
    os.system('shred -u {}'.format(filename))
    os.system(
        'wget -q https://github.com/hse-cs-ami/coursera-intro-dl/releases/download/{} -O {}'.format(path, filename))


class WeekSetup(ABC):

    def __init__(self):
        download_github_code('utils/testing.py')

    @abstractmethod
    def setup(self):
        pass
