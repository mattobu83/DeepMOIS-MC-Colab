IN_DIR="/projects/b1017/Jerry/cancer_subtyping/data/TCGA_LUAD/DGCCA"
OUT_DIR="/projects/b1017/Jerry/cancer_subtyping/results/LUAD/DGCCA"

# Raw data
IN_PATH="/projects/b1017/Jerry/cancer_subtyping/data/TCGA_LUAD/methyl_rnaseq_mirna_minmax1.tsv.gz"
TRAIN_PATH="${IN_DIR}/methyl_rnaseq_mirna_minmax1.train.pkl"
TUNE_PATH="${IN_DIR}/methyl_rnaseq_mirna_minmax1.tune.pkl"
EXP=1

# Some reasonable defaults
#lr=0.005 # Too big a step size for many of the models
lr=0.000001
activation="relu"
bsize=617 # minibatch size
numEpochs=20
truncParam=100 # How many left singular vectors to keep in our data matrices
valfreq=1 # Number of epochs between checking reconstruction error

# Optimizers
# adam_opt="{\"type\":\"adam\",\"params\":{\"adam_b1\":0.1,\"adam_b2\":0.001,\"learningRate\":${lr}}}"
# sgd_momentum_opt="{\"type\":\"sgd_momentum\",\"params\":{\"momentum\":0.99,\"decay\":1.0,\"learningRate\":${lr}}}"
sgd_opt="{\"type\":\"sgd\",\"params\":{\"decay\":1.0,\"learningRate\":${lr}}}"

# Training parameters
architecture="[[19924,500,100],[16630,500,100],[220,500,100]]" # Set architecture of each view
k=100
# rcovStr="0.000001"
# l1=0.0001
# l2=0.01
l1=0.001
l2=0.0001
vnameStr="methylation rnaseq mirna"
# vweights="1.0" # How much to weight each view

# drop spaces
archStr=${architecture// /}
# vweightStr=${vweights// /,}
# rcovStrCat=${rcovStr// /,}

# some modifications
vweightStr="1.0 1.0 1.0"
rcov=0.000001

# Where to write output
BASE="embedding_dgcca_wts=${vweights}_k=${k}_rcov=${rcovStrCat}_arch=${archStr}_l1=${l1}_l2=${l2}_exp=${EXP}"

mkdir "${OUT_DIR}/embeddings"
mkdir "${OUT_DIR}/models"
mkdir "${OUT_DIR}/logs"

EMBEDDINGS_PATH="${OUT_DIR}/embeddings/${BASE}.embedding.npz"
MODEL_PATH="${OUT_DIR}/models/${BASE}.model.npz"
LOG_PATH="${OUT_DIR}/logs/${BASE}.log.txt"
HISTORY_PATH="${OUT_DIR}/logs/${BASE}.history.npz"

echo "python ../dgcca_train_harness.py --input ${IN_PATH} --preptrain ${TRAIN_PATH} --preptune ${TUNE_PATH} --output ${EMBEDDINGS_PATH} --model ${MODEL_PATH} --arch \"${architecture}\" --truncparam ${truncParam} --k ${k} --rcov ${rcovStrCat} --batchSize ${bsize} --l1 ${l1} --l2 ${l2} --vnames ${vnameStr} --activation ${activation} --weights ${vweightStr} --opt \"${sgd_opt}\" --epochs ${numEpochs} --valfreq ${valfreq} --lcurvelog ${HISTORY_PATH} | tee ${LOG_PATH}"
python ../dgcca_train_harness.py --input ${IN_PATH} --preptrain ${TRAIN_PATH} --preptune ${TUNE_PATH} --output ${EMBEDDINGS_PATH} --model ${MODEL_PATH} --arch "${architecture}" --truncparam ${truncParam} --k ${k} --rcov ${rcov} ${rcov} ${rcov} --batchSize ${bsize} --l1 ${l1} --l2 ${l2} --vnames ${vnameStr} --activation ${activation} --weights ${vweightStr} --opt "${sgd_opt}" --epochs ${numEpochs} --valfreq ${valfreq} --lcurvelog ${HISTORY_PATH} | tee ${LOG_PATH}
