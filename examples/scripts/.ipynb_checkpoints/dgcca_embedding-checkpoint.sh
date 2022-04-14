##Directory Path
IN_DIR="/home/pdutta/DGCCA/data/TCGA_BRCA"
OUT_DIR="/home/pdutta/DGCCA/results"

## Raw data
IN_PATH="${IN_DIR}/methyl_rnaseq_mirna_minmax.tsv"
NO_OF_VIEWS=3


## Training parameters
architecture="[[1000,500],[900,400],[200,150]]" # Set architecture of each view
numEpochs=6
latent_dims=100
lr=0.000001
train_batch_size=32
val_batch_size=32
num_workers=$(nproc --all)


## Where to write output
archStr=${architecture// /}
BASE="embedding_views=${NO_OF_VIEWS}_numEpochs=${numEpochs}_val_batch_size=${val_batch_size}__train_batch_size=${train_batch_size}_arch=${archStr}"
mkdir -p "${OUT_DIR}/embeddings"
EMBEDDINGS_PATH="${OUT_DIR}/embeddings/"
LOG_PATH="${OUT_DIR}/logs"



## Run the code
echo "python ../plot_dcca_multi.py --input ${IN_PATH} --n ${NO_OF_VIEWS} --arch \"${architecture}\" --epochs ${numEpochs} --latDim ${latent_dims} --lr ${lr} --log_path ${LOG_PATH} --embedPath ${EMBEDDINGS_PATH} --num_workers ${num_workers} --train_batch_size ${train_batch_size} --val_batch_size ${val_batch_size}"

python ../plot_dcca_multi.py --input ${IN_PATH} --n ${NO_OF_VIEWS} --arch "${architecture}" --epochs ${numEpochs} --latDim ${latent_dims} --lr ${lr} --log_path ${LOG_PATH} --embedPath ${EMBEDDINGS_PATH} --num_workers ${num_workers} --train_batch_size ${train_batch_size} --val_batch_size ${val_batch_size}