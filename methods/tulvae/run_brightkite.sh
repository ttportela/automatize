# Brightkite
TRAIN_FILE="../../data/brightkite/train.csv"
TEST_FILE="../../data/brightkite/test.csv"
EMBEDDINGS_FILE="model/tulvae_brightkite.model"
RESULTS_FILE="results_brightkite.csv"
DATASET_NAME="Brightkite"
LOG_FILE="log_brightkite.txt"
python3 tulvae.py "$TRAIN_FILE" "$TEST_FILE" "$EMBEDDINGS_FILE" "$RESULTS_FILE" "$DATASET_NAME" &> "$LOG_FILE" &
