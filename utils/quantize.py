"""quantizer"""
from math import sqrt
import numpy as np
from utils.onnx_bridge import OnnxBridge
from utils import ans
import pickle
sanity = False
verbose = False
minsize = 512
min_out_channel_size = 1

def compressed_bits(qw):
    """compressed bits"""
    _, c = np.unique(qw, return_counts=True)
    p = c / sum(c)
    return sum(-np.log2(p) * c) + (16 + 16) * len(c)

def measure_cos_err(p, t):
    """measure cos err"""
    assert p.shape == t.shape, 'shapes are  not the same!'
    return p / np.linalg.norm(p) @ t / np.linalg.norm(t)

def measure_cos_sim(p, t, _):
    """measure cos distance"""
    return measure_cos_err(p[0].flatten()[:], t[0].flatten()[:])

def bitstring_to_bytes(s):
    s = '1' + s # to keep leading zeros
    return int(s,2).to_bytes((len(s)+7) // 8, byteorder='big')

def bytes_to_bitstring(bs):
    return  bin(int.from_bytes(bs, byteorder='big')).lstrip('0b')[1:]


def calculate_compressed_size(qwn, emulate_compression=False, save_to=None):
    """calculate_compressed_size"""
    min_out_channel_size = 1
    compressed_size = 0
    riq_dict = {}
    for k, qwt in qwn.items():
        qw, delta = qwt
        shape = qw.shape
        if qw.size > minsize and len(shape) > 1 and shape[0] > min_out_channel_size:
            qw = qw.flatten().astype(np.int32)
            # encoding is expensive.
            # don't do encoding in each step, entropy limit approximation is sufficient
            if emulate_compression:
                compressed_size += compressed_bits(qw)
            else:
                tans = ans.TabledANS.from_data(qw)
                bit_stream = tans.encode_efficient_data(qw)
                if sanity:
                    print("Sanity check for layer ", k)
                    arr = np.array(tans.decode_data(bit_stream))
                    assert np.array_equal(qw, arr)
                compressed_size += len(bit_stream) + tans.total_tables_size
                if save_to != None:
                    riq_dict[k] = (tans, bitstring_to_bytes(bit_stream), delta, shape)

        else:
            compressed_size += 32 * qw.size
            riq_dict[k] = (None, qw * delta, 1.0, shape)
    if save_to != None:
        with open(save_to, 'wb') as fn:
            pickle.dump(riq_dict, fn)

    return compressed_size

def get_quantized_model(
        model_fn,
        calibration_dataset,
        eps=0.01,
        distortion=0.005,
        skip_first=False,
        compare_function=measure_cos_sim,
        save_to=None):
    """should return quantized model
    do quantization and print quantization logging"""
    err_thr = 1.0 - distortion
    ob = OnnxBridge(model_fn)

    min_out_channel_size = 1
    original_outs = [ob(i) for i in calibration_dataset]
    ws = ob.get_weights()
    qws = {}
    qwn = {}
    max_dim = 0

    prod_norm = 1.0
    model_numel = 0
    for k, w in ws.items():
        if w.size > minsize and len(w.shape) > 1 and w.shape[0] > min_out_channel_size:
            model_numel += w.size
            prod_norm *= sqrt((w ** 2).sum())
        if w.size > max_dim:
            max_dim = w.size

    upper_bound = sqrt(max_dim / 24.0) / (sqrt(eps) * eps)  # (sqrt(eps)*eps)
    lower_bound = sqrt(max_dim / 24.0) / (1 - eps)  # (1-eps)
    step = sqrt(upper_bound - lower_bound)
    print("searching for k in the range [", lower_bound, ",", upper_bound, "] with steps of: ", step)

    idx = 0
    k_const = lower_bound
    while (step > 3 and k_const <= upper_bound):
        skip_f = skip_first
        w_size = 0
        compressed_size = 0
        stat = []
        epsilons = []

        ##################################################
        for k, w in ws.items():
            w_size += 32 * w.size
            if skip_f:
                qws[k] = w.copy()
                qwn[k] = (w.copy(), 1.0)
                skip_f = False
                continue

            if w.size > minsize and len(w.shape) > 1 and w.shape[0] > min_out_channel_size:
                delta = np.linalg.norm(w) / k_const + np.linalg.norm(w) * eps \
                * sqrt(24 / w.size)

                epsilons += [(eps + sqrt(w.size / 24) / k_const) ** 2]
                stat += [np.ceil((w.max() - w.min()) / delta)]
                qw = np.round(w / delta).flatten().astype(np.int32)
                qws[k] = qw.reshape(w.shape) * delta
                qwn[k] = (qw.reshape(w.shape), delta)

            else:
                qws[k] = w.copy()
                qwn[k] = (w.copy(), 1.0)
        ##################################################################
        ob.set_weights(qws)
        outs = [ob(i) for i in calibration_dataset]

        mean_err = 0
        for orig_output, quant_output, inputs in zip(original_outs, outs, calibration_dataset):
            mean_err += compare_function(orig_output, quant_output, inputs)

        mean_err /= len(outs)
        if mean_err > err_thr:
            print("k = ", end="")
            print('%0.2f' % k_const, end="")
            print(" complies with the distortion constraint", distortion, end="")
            print(". Approximated CR: ", w_size / calculate_compressed_size(qwn, True))
            if verbose:
                print("err: ", 1 - mean_err)
                print("step: ", step)
                print("number of bins:")
                print(stat)
                print("mean rate:")
                print(sum([np.log2(s) for s in stat]) / len(stat))
                print("epsilons:")
                print(epsilons)
                print("mean epsilon:")
                print(sum(epsilons) / len(epsilons))
                print("mean_err:/out_err")
                print((sum(epsilons) / len(epsilons)) / (1 - mean_err))
            upper_bound = k_const
            step = sqrt(step)
            lower_bound = max(lower_bound, k_const - step * np.floor(step))
            k_const = lower_bound
            idx = 0
            if step > 3:
                print("searching for k in the range [", lower_bound, ",", upper_bound, "] with steps of: ", step)


        else:
            idx += 1
            k_const = lower_bound + idx * step
    # For quantize model that fits constraint, do encoding to measure actual compression rate
    print("Start compressing with ANS encoder..." )
    print("ANS achieved CR: ", w_size / calculate_compressed_size(qwn, False, save_to))
    return ob

def get_quantized_model_by_const(model_fn, const):
    """get quantized model by const"""
    ob = OnnxBridge(model_fn)
    min_out_channel_size = 1
    ws = ob.get_weights()
    qws = {}
    qwn = {}
    eps = 0.01
    n_const = const
    w_size = 0
    minsize = 10

    ##################################################
    for k, w in ws.items():
        w_size += 32 * w.size
        if w.size > minsize and len(w.shape) > 1 and w.shape[0] > min_out_channel_size:
            delta = np.linalg.norm(w) / n_const + np.linalg.norm(w) * eps * sqrt(24 / w.size)
            qw = np.round(w / delta).flatten().astype(np.int32)
            qws[k] = qw.reshape(w.shape) * delta
            qwn[k] = qw.reshape(w.shape)
        else:
            qws[k] = w.copy()
            qwn[k] = w.copy()

    ob.set_weights(qws)
    print("ANS achieved CR: ", w_size / calculate_compressed_size(qwn, False))

    return ob
