import os
import torch
import torchaudio


class AudioClassificationDataset(torch.utils.data.Dataset):
    label_to_class = ['BOOK', 'EAT', 'JOKE', 'MOTIVATE', 'PRAISE']
    label_to_text = ['посоветуй книжку', 'что бы мне съесть', 'расскажи шутку',
                     'мотивируй на работу', 'похвали меня']
    class_to_label = {class_name: label for label, class_name in enumerate(label_to_class)}

    def __init__(self, root, sample_rate=16000, transform=None):
        self.root = root
        self.sample_rate = sample_rate
        self.transform = transform
        self.files = sorted(os.listdir(self.root))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename = self.files[index]
        classname = filename.split('_')[1]
        label = self.class_to_label[classname]

        filename = os.path.join(self.root, self.files[index])
        waveform, rate = torchaudio.load(filename)
        assert rate == self.sample_rate, 'Частота дискретизации файла не совпадает с указанной!'

        waveform = waveform.squeeze()
        if self.transform is not None:
            waveform = self.transform(waveform)

        return waveform, label
