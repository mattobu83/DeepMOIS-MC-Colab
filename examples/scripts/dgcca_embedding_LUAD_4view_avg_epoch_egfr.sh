start_epoch=1
end_epoch=5
for ((epoch=$start_epoch; epoch<=$end_epoch; epoch++))
do
  echo "Running Python script for epoch $epoch"
  CANCER_TYPE="LUAD"
  ##Directory Path
  IN_DIR="/content/drive/MyDrive/TCGA_LUAD_DGCCA"
  OUT_DIR="./results/${CANCER_TYPE}"
  
  
  ## Raw data
  IN_PATH="${IN_DIR}/methyl_rnaseq_mirna_img_minmax_avg_egfr.tsv"
  NO_OF_VIEWS=4
  
  
  ## Training parameters
  architecture="[[1000,500],[900,400],[200,150],[15,10]]" # Set architecture of each view
  #numEpochs=5
  latent_dims=100
  lr=0.0000001
  train_batch_size=32
  val_batch_size=32
  num_workers=2
  
  
  ## Where to write output
  archStr=${architecture// /}
  BASE="views=${NO_OF_VIEWS}_numEpochs=${epoch}__val_batch_size=${val_batch_size}__train_batch_size=${train_batch_size}__arch=${archStr}__lr=${lr}__latDim=${latent_dims}"
  EMBEDDINGS_PATH="${OUT_DIR}/embeddings/${BASE}"
  #rm -r "${EMBEDDINGS_PATH}"
  mkdir -p "${EMBEDDINGS_PATH}"
  LOG_PATH="${OUT_DIR}/logs/${CANCER_TYPE}"
  MODEL_PATH="${OUT_DIR}/saved_models/${BASE}"
  FINAL_EMBEDDING_PATH="./results/final_embedding/${CANCER_TYPE}/${BASE}"
  mkdir -p "${LOG_PATH}"
  
  
  ## Run the code
  echo "python /content/DeepMOIS-MC/examples/dcca_multi.py --input ${IN_PATH} --n ${NO_OF_VIEWS} --arch \"${architecture}\" --epochs ${epoch} --latDim ${latent_dims} --lr ${lr} --log_path ${LOG_PATH} --model_path ${MODEL_PATH} --embedPath ${EMBEDDINGS_PATH} --num_workers ${num_workers} --train_batch_size ${train_batch_size} --val_batch_size ${val_batch_size} --base_name ${BASE} --cancer_type ${CANCER_TYPE} --final_embed_path ${FINAL_EMBEDDING_PATH}"
  
  # --train_batch_size ${train_batch_size} --val_batch_size ${val_batch_size}
  python /content/DeepMOIS-MC/examples/dcca_multi4_egfr.py --input ${IN_PATH} --n ${NO_OF_VIEWS} --arch "${architecture}" --epochs ${epoch} --latDim ${latent_dims} --lr ${lr} --log_path ${LOG_PATH} --model_path ${MODEL_PATH} --embedPath ${EMBEDDINGS_PATH} --num_workers ${num_workers} --base_name ${BASE} --cancer_type ${CANCER_TYPE} --final_embed_path ${FINAL_EMBEDDING_PATH} | tee $LOG_PATH"/"$(date "+%Y-%m-%d_%H:%M:%S")"_log.txt"
done
