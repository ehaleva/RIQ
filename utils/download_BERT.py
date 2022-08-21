""" download BERT model from Neural Magic Sparse Zoo"""
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sparsezoo.models import Zoo

stub = "zoo:nlp/question_answering/distilbert-none/pytorch/huggingface/squad/base-none"
override_folder_name = 'sparsezoo_bert'
model = Zoo.download_model_from_stub(stub, override_folder_name=override_folder_name,
                                     override_parent_path="models/downloads")
os.system('mv models/downloads/sparszaoo_bert/model.onnx models/BERT.onnx')
os.system('rm -df models/downloads')
