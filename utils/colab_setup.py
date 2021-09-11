import os
import shutil
from zipfile import ZipFile
from abc import ABC
from abc import abstractmethod


def download_github_code(path):
    filename = path.rsplit('/')[-1]
    os.system('shred -u {}'.format(filename))
    os.system(
        'wget -q https://raw.githubusercontent.com/hse-cs-ami/coursera-advanced-dl/main/{} -O {}'.format(path, filename))


def download_github_release(path):
    filename = path.rsplit('/')[-1]
    os.system('shred -u {}'.format(filename))
    os.system(
        'wget -q https://github.com/hse-cs-ami/coursera-advanced-dl/releases/download/{} -O {}'.format(path, filename))


def download_github_raw(path):
    filename = path.rsplit('/')[-1]
    os.system('shred -u {}'.format(filename))
    os.system(
        'https://github.com/hse-cs-ami/coursera-advanced-dl/blob/main/{}?raw=true -O {}'.format(path, filename))


class WeekSetup(ABC):
    def __init__(self):
        download_github_code('utils/testing.py')

    @abstractmethod
    def setup(self):
        pass


class Audio(WeekSetup):
    def __init__(self):
        super().__init__()
        if os.path.isdir('week03'):
            shutil.rmtree('week03')

        os.mkdir('week03')
        code_base = {'asr': ['alphabet.py', 'metrics.py', 'model.py'],
                     'cls': ['dataset.py']}

        for code_dir, files in code_base.items():
            os.mkdir(os.path.join('week03', code_dir))
            for filename in files:
                github_path = os.path.join('week03-dl-audio', code_dir, filename)
                local_path = os.path.join('week03', code_dir, filename)
                download_github_code(github_path)
                os.rename(filename, local_path)

        download_github_raw(os.path.join('week03', 'data', 'audio-dataset.zip'))

    def setup(self):
        os.system('pip install torchaudio editdistance')
        with ZipFile('audio-dataset.zip', 'r') as zip_ref:
            zip_ref.extractall()
