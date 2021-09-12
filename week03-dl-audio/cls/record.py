import base64
from google.colab import output
from ipywidgets import widgets
from note_seq.audio_io import wav_data_to_samples_pydub
from traitlets import traitlets
from IPython.display import display, Javascript


def record(seconds=3, sample_rate=16000, normalize_db=0.1):
    """
    Record audio via microphone with Javascript
    Based on https://github.com/magenta/ddsp/blob/main/ddsp/colab/colab_utils.py
    """
    record_js_code = """
    const sleep  = time => new Promise(resolve => setTimeout(resolve, time))
    const b2text = blob => new Promise(resolve => {
    const reader = new FileReader()
        reader.onloadend = e => resolve(e.srcElement.result)
        reader.readAsDataURL(blob)
    })
    var record = time => new Promise(async resolve => {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        recorder = new MediaRecorder(stream)
        chunks = []
        recorder.ondataavailable = e => chunks.push(e.data)
        recorder.start()
        await sleep(time)
        recorder.onstop = async ()=>{
            blob = new Blob(chunks)
            text = await b2text(blob)
            resolve(text)
        }
        recorder.stop()
    })
    """

    print('Starting recording for {} seconds...'.format(seconds))
    display(Javascript(record_js_code))
    audio_string = output.eval_js('record(%d)' % (seconds * 1000.0))
    print(audio_string)
    print('Finished recording!')
    audio_bytes = base64.b64decode(audio_string.split(',')[1])
    return wav_data_to_samples_pydub(wav_data=audio_bytes, sample_rate=sample_rate,
                                     normalize_db=normalize_db, num_channels=1)


class LoadedButton(widgets.Button):
    def __init__(self, audio=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_traits(audio=traitlets.Any(audio))


def record_audio(button):
    button.audio = record()
