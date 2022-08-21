"""evaluate nlp"""
import sys
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from collections import Counter
import re
import string
import numpy as np
import torch
from transformers import DistilBertTokenizer
from transformers.data.processors.squad import SquadV1Processor
from utils.onnx_bridge import OnnxBridge

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')

def get_prediction(model, qid, size_input):
    """get prediction"""
    question = examples[qid_to_example_index[qid]].question_text
    context = examples[qid_to_example_index[qid]].context_text
    inputs = tokenizer(question, context, return_tensors='pt')
    size = dict(inputs)['input_ids'][0].size()[0]
    if size > size_input:
        input_ids = dict(inputs)['input_ids'][0, :size_input]
        attention_mask = dict(inputs)['attention_mask'][0, : size_input]
    else:
        input_ids = torch.cat([dict(inputs)['input_ids'][0],
                               torch.zeros(size_input - size, dtype=torch.int64)])
        attention_mask = torch.cat(
            [dict(inputs)['attention_mask'][0], torch.zeros(size_input - size, dtype=torch.int64)])
    inputs2 = [np.array(torch.reshape(input_ids, (1, size_input), )),
               np.array(torch.reshape(attention_mask, (1, size_input)))]

    outputs = model(inputs2)

    answer_start = torch.argmax(torch.Tensor(outputs[0]))

    answer_end = torch.argmax(torch.Tensor(outputs[1])) + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

    return answer


def normalize_text(s):
    """normalize text"""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def remove_accents(old):
        new = re.sub(r'[àáâãäåạằ]', 'a', old)
        new = re.sub(r'[èéêë]', 'e', new)
        new = re.sub(r'[ìíîï]', 'i', new)
        new = re.sub(r'[òóôõö]', 'o', new)
        new = re.sub(r'[ùúûü]', 'u', new)
        new = re.sub(r'[ć]', 'c', new)
        new = re.sub(r'[ś]', 's', new)
        new = re.sub(r'[ñńǹň]', 'n', new)
        return new

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(remove_accents(lower(s)))))


def get_gold_answers(example):
    """get gold answers"""
    gold_answers = \
        [tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(
                tokenizer(answer["text"], "", return_tensors='pt')['input_ids'][0][1:-2]))
            for answer in example.answers if answer["text"]]
    if not gold_answers:
        gold_answers = [""]
    return gold_answers


def compute_exact_match(prediction, truth):
    """compute exact match"""
    return int(normalize_text(prediction) == normalize_text(truth))


def compute_f1(prediction, truth):
    """compute f1"""
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = Counter(pred_tokens) & Counter(truth_tokens)

    if sum(common_tokens.values()) == 0:
        return 0

    prec = sum(common_tokens.values()) / len(pred_tokens)
    rec = sum(common_tokens.values()) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)


def main():
    """main"""
    global examples
    global qid_to_example_index
    model_path = sys.argv[1]
    dataset_path = sys.argv[2]
    model = OnnxBridge(model_path)
    dims = model.get_inputs()
    size_input = dims[1][1]
    em_score = 0
    f1_score = 0
    total = 0
    processor = SquadV1Processor()
    examples = processor.get_dev_examples(dataset_path, filename="dev-v1.1.json")
    qid_to_example_index = {example.qas_id: i for i, example in enumerate(examples)}
    qid_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
    answer_qids = [qas_id for qas_id, has_answer in qid_to_has_answer.items() if has_answer]

    with torch.no_grad():
        for i in answer_qids:
            example = examples[qid_to_example_index[i]]
            prediction = get_prediction(model, i, size_input)
            gold_answers = get_gold_answers(example)
            em_score += max(compute_exact_match(prediction, answer) for answer in gold_answers)
            f1_score += max(compute_f1(prediction, answer) for answer in gold_answers)
            total += 1

    print('f1=', f1_score / total)
    print('em=', em_score / total)


main()
