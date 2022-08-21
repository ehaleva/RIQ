
usage() {
    echo "evaluate_quantization.sh MODEL VALIDATION_DATASET [-d DISTORTION] [-c CALIBRATION_DATA]" 
    echo "                         MODEL can be either one of the following models: VGG, resnet, alexnet, ViT, YOLO, or BERT"
    echo "                               alternatively an onnx model filename"
    echo "                         VALIDATION_DATASET is a path to an Imagenet folder for the case of VGG, resnet, alexnet or ViT"
    echo "                                               a path to a COCO folder for the case of YOLO"
    echo "                                               a path to a XXX  folder for the case of BERT"
    echo "                         DISTORTION is a cosine similarity constraint for calibration. a small value << 1. Default=0.005"
    echo "                         CALIBRATION_DATA is a set of samples that represent the data. Default means random data will be generated"
    echo "                                          note that BERT quantization does not support a calibration dataset"
    exit -1
}

checkopts()
{
  DISTORTION=0.005
  mkdir -p empty_calibration
  CAL_DATASET=$(pwd)/empty_calibration
  while getopts 'd:c:' opt
  do
    case "${opt}" in
      d)
        DISTORTION=$OPTARG
        ;;
      c)
        CAL_DATASET=$OPTARG
        ;;
      *)
        echo "Unknown option ${opt}!"
        display_usage
        exit 1
    esac
  done
}

VAL_DATASET=$(realpath $2)

if (($# < 2)); then
    usage
fi

checkopts ${@:3}


if [[ -f $1 ]]; then
    MODEL_ONNX=$1
    MODEL=`basename ${MODEL_ONNX} .onnx`
else
    MODEL=$1
    if  [[ $1 == "ViT" ]] || [[ $1 == "alexnet" ]] || [[ $1 == "resnet" ]] || [[ $1 == "VGG" ]] || [[ $1 == "YOLO" ]] || [[ $1 == "BERT" ]]; then
	MODEL_ONNX=models/$1.onnx
        if [[ ! -f "$MODEL_ONNX" ]]; then
            echo "Downloading pretrained $MODEL, this might take a while"
            python utils/download_$MODEL.py || exit 1
	fi
    fi
fi

	   
if [[ ! -f "$MODEL_ONNX" ]]; then
    echo "Does not support model $1 or Could not find valid ONNX file in path:" $1
    usage
fi

echo 'model:' $MODEL 
echo 'onnx file name:' $MODEL_ONNX
echo 'validation dataset:' $VAL_DATASET
echo 'calibration dataset:'  $CAL_DATASET
echo 'distortion:' ${DISTORTION}

if [[ ! -d "$CAL_DATASET" ]]; then
    echo "Could not find valid CALIBRATION_DATASET directory in path:" $CAL_DATASET
    usage
fi


mkdir -p logs
if (($# > 3)); then
  LOG_FILE=logs/log_${MODEL}_${DISTORTION}.txt
else
  LOG_FILE=logs/log_${MODEL}_${DISTORTION}_random.txt
fi
LOG_FILE=$(realpath $LOG_FILE)
MODEL_ONNX=$(realpath $MODEL_ONNX)

if  [[ $MODEL == *"VGG"* ]] || [[ $MODEL == *"resnet"* ]] || [[ $MODEL == "ViT" ]] || [[ $MODEL == "alexnet" ]]; then
    if [[ $DISTORTION == "All" ]]; then
	echo "Calling combined python script"
        python python_files/resnet_vgg/run_all_distortions.py $MODEL $MODEL_ONNX $VAL_DATASET ${CAL_DATASET[*]} | tee ${LOG_FILE}
    else
        python compare_cv.py $MODEL $MODEL_ONNX $VAL_DATASET ${CAL_DATASET[*]} $DISTORTION| tee ${LOG_FILE} || exit 1
        echo "To summarize:"
        awk '/Actual CR:/ {print; }' ${LOG_FILE} >> ${LOG_FILE}
    fi
elif [[ $MODEL == *"YOLO"* ]]; then
    if [[ ! -d third_party/yolov5 ]]; then
	echo "cloning third party repo Ultralytics"
        mkdir -p third_party/datasets
	cd third_party
	git clone https://github.com/ultralytics/yolov5.git
	cd -
    fi
    PATH_QUANT_MODEL_ONNX=$(realpath 'models/'$MODEL'_quant.onnx')
    #python utils/quantize_yolo.py $MODEL_ONNX $PATH_QUANT_MODEL_ONNX 1663 | tee ${LOG_FILE}
    python utils/quantize_yolo.py $MODEL_ONNX $PATH_QUANT_MODEL_ONNX $DISTORTION $CAL_DATASET | tee ${LOG_FILE}
    cd third_party
    mkdir -p datasets
    rm datasets/coco
    ln -s $VAL_DATASET datasets/coco 
    echo "Evaluating Quantized model" >> ${LOG_FILE}
    python yolov5/val.py --weights $PATH_QUANT_MODEL_ONNX --data yolov5/data/coco.yaml | tee -a ${LOG_FILE}
    echo "Evaluating Original model" >> ${LOG_FILE}
    python yolov5/val.py --weights $MODEL_ONNX --data yolov5/data/coco.yaml | tee -a ${LOG_FILE}
    cd -
elif [[ "${MODEL,,}" == *"bert"* ]]; then
    if [[ $CAL_DATASET != $(pwd)/empty_calibration ]]; then
        echo "BERT quantization does not supoort a calibration dataset"
	usage
    fi
    PATH_QUANT_MODEL_ONNX=$(realpath 'models/'$MODEL'_quant.onnx')
    python utils/quantize_bert.py $MODEL_ONNX $PATH_QUANT_MODEL_ONNX $DISTORTION $VAL_DATASET | tee ${LOG_FILE}
    echo "Evaluating Quantized model" >> ${LOG_FILE}
    python evaluate_nlp.py $PATH_QUANT_MODEL_ONNX $VAL_DATASET | tee -a ${LOG_FILE}
    echo "Evaluating Original model" >> ${LOG_FILE}
    python evaluate_nlp.py $MODEL_ONNX $VAL_DATASET | tee -a ${LOG_FILE}
else
    usage
fi
#!/bin/bash
