import os
import random
import torch
import numpy as np
from pydub import AudioSegment


class VoiceAssistant(object):
    label_to_class = ['BOOK', 'EAT', 'JOKE', 'MOTIVATE', 'PRAISE']
    sample_rate = 48000

    def __init__(self, cls_model, spectrogramer, device):
        self.cls_model = cls_model.eval()
        self.spectrogramer = spectrogramer
        self.device = device

        self.answers = [[]] * len(self.label_to_class)
        for i, class_name in enumerate(self.label_to_class):
            files = os.listdir(os.path.join('voice-assistant', class_name))
            for file_name in files:
                file_path = os.path.join('voice-assistant', class_name, file_name)
                audio = AudioSegment.from_file(file_path)
                audio = np.array(audio.get_array_of_samples())
                self.answers[i] += [audio]

    def answer(self, waveform):
        with torch.no_grad():
            spec = self.spectrogramer(waveform.to(self.device).unsqueeze(0))
            logits = self.cls_model(spec).cpu().squeeze(0)
            label = torch.argmax(logits).item()

        return random.choice(self.answers[label])
