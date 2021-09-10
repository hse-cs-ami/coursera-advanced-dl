import os
from zipfile import ZipFile
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


class Audio(WeekSetup):
    def __init__(self):
        super().__init__()
        if os.path.isdir('week03'):
            os.rmdir('week03')

        os.mkdir('week03')
        code_base = {'asr': ['alphabet.py', 'metrics.py', 'model.py'],
                     'cls': ['dataset.py']}

        for code_dir, files in code_base.items():
            os.mkdir(os.path.join('week03', code_dir))
            for filename in files:
                path = os.path.join('week03', code_dir, filename)
                download_github_code(path)
                os.rename(filename, path)

        download_github_code(os.path.join('week03', 'data', 'audio-dataset.zip'))

    def setup(self):
        os.system('pip install torchaudio editdistance')
        with ZipFile('audio-dataset.zip', 'r') as zip_ref:
            zip_ref.extractall()
