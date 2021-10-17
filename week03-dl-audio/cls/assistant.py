import os
import random
import torch
from pydub import AudioSegment


class VoiceAssistant(object):
    label_to_class = ['BOOK', 'EAT', 'JOKE', 'MOTIVATE', 'PRAISE']

    def __init__(self, cls_model, spectrogramer):
        self.cls_model = cls_model.eval()
        self.spectrogramer = spectrogramer

        self.answers = [[]] * len(self.label_to_class)
        for i, class_name in enumerate(self.label_to_class):
            files = os.listdir(os.path.join('voice-assistant', class_name))
            for file_name in files:
                file_path = os.path.join('voice-assistant', class_name, file_name)
                self.answers[i] += [AudioSegment.from_file(file_path)]

    def answer(self, waveform):
        with torch.no_grad():
            spec = self.spectrogramer(waveform.unsqueeze(0))
            logits = self.cls_model(spec).squeeze(0)
            label = torch.argmax(logits).item()

        return random.choice(self.answers[label])
