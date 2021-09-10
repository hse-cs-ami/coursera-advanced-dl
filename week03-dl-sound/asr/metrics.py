import torch
import editdistance
from collections import namedtuple


ASRMetrics = namedtuple('ASRMetrics', ['CER', 'WER'])


def asr_metrics(predicts, targets):
    cer, wer = [], []
    for predict_string, target_string in zip(predicts, targets):
        predict_words = predict_string.split(' ')
        target_words = target_string.split(' ')

        dist_char = editdistance.eval(predict_string, target_string)
        dist_word = editdistance.eval(predict_words, target_words)

        cer.append(dist_char / len(target_string))
        wer.append(dist_word / len(target_words))

    metrics = ASRMetrics(CER=torch.tensor(cer).mean().item(), WER=torch.tensor(wer).mean().item())
    return metrics
