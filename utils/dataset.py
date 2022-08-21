"""prepare calibration dataset for quantization"""
import numpy as np
import torch as pt
from transformers import DistilBertTokenizer
from transformers.data.processors.squad import SquadV1Processor
from utils.onnx_bridge import OnnxBridge
from utils.image_dir import ImageDir


def prepare_dataset_images(dataset_path, path_ob):
    """prepare dataset images"""
    ob = OnnxBridge(path_ob)
    dims = ob.get_inputs()

    image_dims = tuple(dims[0])

    images = ImageDir(dataset_path, image_dims)
    _images = []
    if dims[1:] != []:
        pt.manual_seed(20)
        for im_idx in range(len(images)):
            rand_inp = [images[im_idx]]
            for el in dims[1:]:
                rand_inp.append(10 * pt.randn(tuple(el)).numpy())
            _images.append(rand_inp)
        images = _images  # [0]
    if len(images) != 0:
        print("Using Calibration set with", len(images), "sample images")
    else:
        print("Using Calibration set with randomly generated input")
        images = []
        for i in range(3):
            images.append(np.random.normal(size=tuple(dims[0])).astype(np.float32))

    return images


def calibration_equal_size(size_text, size_input):
    size_equals_size_input = True
    for k in size_text:
        if k != size_input:
            size_equals_size_input = False
    return size_equals_size_input


def prepare_dataset_text(dataset_path, size_input=384):
    """Choose 3 samples from the validation dataset for calibration
    best performance is if we choose text which is similar in length to the model's input size"""
    processor = SquadV1Processor()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
    examples = processor.get_dev_examples(dataset_path, filename="dev-v1.1.json")
    qid_to_example_index = {example.qas_id: i for i, example in enumerate(examples)}
    qid_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
    answer_qids = [qas_id for qas_id, has_answer in qid_to_has_answer.items() if has_answer]
    size_text = np.array([0, 0, 0])
    text = np.array([None, None, None])
    if len(answer_qids) >= 3:
        for i in range(3):
            question = examples[qid_to_example_index[answer_qids[i]]].question_text
            context = examples[qid_to_example_index[answer_qids[i]]].context_text
            inputs = tokenizer.encode_plus(question, context, return_tensors='pt')
            size = dict(inputs)['input_ids'][0].size()[0]
            text[i] = inputs
            size_text[i] = size

        for i in range(len(answer_qids)):

            if calibration_equal_size(size_text, size_input):
                return text
            question = examples[qid_to_example_index[answer_qids[i]]].question_text
            context = examples[qid_to_example_index[answer_qids[i]]].context_text
            inputs = tokenizer.encode_plus(question, context, return_tensors='pt')
            size = dict(inputs)['input_ids'][0].size()[0]
            if abs(size - size_input) < np.max(abs(size_text - size_input)):
                j = np.argmax(abs(size_text - size))
                if size > size_input:
                    input_ids = dict(inputs)['input_ids'][0, :size_input]
                    attention_mask = dict(inputs)['attention_mask'][0, : size_input]
                else:
                    input_ids = pt.cat([dict(inputs)['input_ids'][0],
                                        pt.zeros(size_input - size, dtype=pt.int64)])
                    attention_mask = pt.cat([dict(inputs)['attention_mask'][0],
                                             pt.zeros(size_input - size, dtype=pt.int64)])
                inputs = [np.array(pt.reshape(input_ids, (1, size_input), )),
                          np.array(pt.reshape(attention_mask, (1, size_input)))]
                text[j] = inputs
                size_text[j] = size

        print("Created a Calibration set with", len(text), "text samples of lengths", size_text)
    else:
        text = []
        for i in range(3):
            input_ids = np.random.randint(20000, size=(1, size_input), dtype=np.int64)
            attention_mask = np.ones((1, size_input), dtype=np.int64)
            text.append((input_ids, attention_mask))
    return text
