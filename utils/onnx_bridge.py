"""onnx bridge"""
import numpy as np
import onnx
import onnx.numpy_helper as nh
import onnxruntime
import torch

class OnnxBridge:
    """class onnxBridge"""
    def __init__(self, fn):
        self.m = onnx.load(fn)
        cuda = torch.cuda.is_available()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
            if cuda else ['CPUExecutionProvider']
        self.sess = onnxruntime.InferenceSession(fn, providers=providers)

    def save(self, fn):
        onnx.save(self.m, fn)

    def __call__(self, x):
        if not self.sess:
            cuda = torch.cuda.is_available()
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                if cuda else ['CPUExecutionProvider']
            self.sess = onnxruntime.InferenceSession(self.m.SerializeToString(),
                                                     providers=providers)

        names = [i.name for i in self.sess.get_inputs()]
        if not type(x) in (list, tuple):
            x = [x]

        return self.sess.run(None, {name:v for name, v in zip(names, x)})

    def get_weights(self):
        return {(i, w.name): nh.to_array(w).copy() for i, w in enumerate(self.m.graph.initializer)}

    def set_weights(self, weights):
        assert len(weights) == len(self.m.graph.initializer), 'incorrect number of weights!'
        for k, w in weights.items():
            i, name = k
            if self.m.graph.initializer[i].data_type == 1:
                self.m.graph.initializer[i].CopyFrom(nh.from_array(np.float32(w), name=name))
        self.sess = None

    def get_inputs(self):
        inputs = []
        for inp in self.m.graph.input:
            tensor_type = inp.type.tensor_type
            if tensor_type.HasField("shape"):
                # iterate through dimensions of the shape:
                t_inp = []
                for d in tensor_type.shape.dim:
                    t_inp.append(d.dim_value)
                    inputs.append(t_inp)
            else:
                raise ValueError("Inputs shape not found")
        return inputs
