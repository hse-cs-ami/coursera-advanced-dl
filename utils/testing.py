import json
import requests
import torch


class Grader:

    def __init__(self, assignment_key, parts=()):
        self.submission_page = 'https://www.coursera.org/api/onDemandProgrammingScriptSubmissions.v1'
        self.assignment_key = assignment_key
        self.answers = {part: None for part in parts}

    def submit(self, email, token):
        submission = {
            'assignmentKey': self.assignment_key,
            'submitterEmail': email,
            'secret': token,
            'parts': {}
        }
        for part, output in self.answers.items():
            if output is not None:
                submission['parts'][part] = {
                    'output': output
                }
            else:
                submission['parts'][part] = dict()

        request = requests.post(self.submission_page, data=json.dumps(submission))
        response = request.json()
        if request.status_code == 201:
            print('Submitted to Coursera platform. See results on assignment page!')
        elif u'details' in response and u'learnerMessage' in response[u'details']:
            print(response[u'details'][u'learnerMessage'])
        else:
            print('Unknown response from Coursera: {}'.format(request.status_code))
            print(response)

    def set_answer(self, part, answer):
        if isinstance(answer, str):
            self.answers[part] = answer
        else:
            try:
                self.answers[part] = ' '.join(map(repr, answer))
            except TypeError:
                self.answers[part] = repr(answer)


class Tester:

    def __init__(self, assignment_key, parts=()):
        self.grader: Grader = Grader(assignment_key=assignment_key, parts=parts)
        self.email: str = ''
        self.token: str = ''
        self.parts = parts

    def set_email(self, email: str):
        self.email = email

    def set_token(self, token: str):
        self.token = token


class TestWeek01(Tester):
    pass


class TestWeek02(Tester):
    pass


class TestWeek03(Tester):

    def test01(self, model_class):
        pass
        # TODO: test model output shape

    def test02(self, cls_model, spectrogramer, test_loader, device):
        accuracy = 0.0
        for waveforms, labels in test_loader:
            waveforms, labels = waveforms.to(device), labels.to(device)

            with torch.no_grad():
                specs = spectrogramer(waveforms)
                logits = cls_model(specs)

            accuracy += (logits.argmax(dim=1) == labels).sum().item()

        accuracy /= len(test_loader.dataset)
        # TODO: test accuracy value

    def test03(self, cer, wer):
        pass
        # TODO: test cer and wer values


class TestWeek04(Tester):
    pass


class TestWeek05(Tester):
    pass
