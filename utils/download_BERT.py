""" download BERT model from Neural Magic Sparse Zoo"""
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sparsezoo import Model

stub = "zoo:nlp/question_answering/distilbert-none/pytorch/huggingface/squad/base-none"
download_path = 'models/downloads/sparsezoo_bert'
model = Model(stub, download_path=download_path)
os.system('mv models/downloads/sparszaoo_bert/model.onnx models/BERT.onnx')
os.system('rm -df models/downloads')
