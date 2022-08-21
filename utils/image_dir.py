"""prepare images for calibration dataset"""
import glob
import cv2
import numpy as np

class ImageDir:
    """image dir"""
    def __init__(self, dataset_calibration_path, dims=(1, 3, 512, 512)):
        self.dataset_calibration_path = dataset_calibration_path
        self.dims = dims
        self.additional_input_dims = dims[1:]
        self.fns = glob.glob(self.dataset_calibration_path, recursive=True)
        self.originals = [cv2.imread(fn) for fn in self.fns]
        def process(i):
            i = cv2.resize(i, (self.dims[2], self.dims[2]))
            i = (np.float32(i)/255-0.5)*2
            if self.dims[1] == 3:
                i = i.transpose(2, 0, 1).reshape(1, 3, i.shape[1], i.shape[0])
            else:
                i = i.reshape(1, i.shape[0], i.shape[1], i.shape[2])
            return i
        self.processed = [process(o) for o in self.originals]

    def __len__(self):
        return len(self.originals)

    def __getitem__(self, ix):
        if self.dims[1] == 3 or self.dims[-1] == 3:
            return self.processed[ix].copy()
        elif self.dims[1] == 6:
            return hstack((self.processed[ix], self.processed[(ix+0)%len(self.processed)]))
        else:
            raise ValueError("ImageDir: model requires image dim that is not supported.")

