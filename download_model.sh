%%bash
cd "$(dirname "${0}")"
BASE_DIR="$(pwd)"

BASE_DIR="$(pwd)"
MODELS_DIRECTORY=models
MODEL_VERSION=model_v5

MODEL_DIRECTORY="${MODELS_DIRECTORY}"

MODEL_NAME=pytorch-gpt2-xl-aid2-v5
MODEL_TORRENT_URL="https://raw.githubusercontent.com/AccidentallyOnPurpose/pytorch-AIDungeon/f692e39d84b21d39da9819142165a05a03030892/generator/gpt2/models/model_v5_pytorch.torrent"
MODEL_TORRENT_BASENAME="$(basename "${MODEL_TORRENT_URL}")"

download_torrent() {
  echo "Creating directories."
  mkdir -p "${MODEL_DIRECTORY}"
  cd "${MODEL_DIRECTORY}"
  wget "${MODEL_TORRENT_URL}"
  which aria2c > /dev/null
  if [ $? == 0 ]; then
    echo -e "\n\n==========================================="
    echo "We are now starting to download the model."
    echo "It will take a while to get up to speed."
    echo "DHT errors are normal."
    echo -e "===========================================\n"
    echo "$MODEL_TORRENT_BASENAME"
    echo "${MODEL_TORRENT_BASENAME%.*}"
    ls
    aria2c \
      --max-connection-per-server 16 \
      --split 64 \
      --bt-max-peers 500 \
      --seed-time=0 \
      --summary-interval=15 \
      --disable-ipv6 \
      "${MODEL_TORRENT_BASENAME}"
    echo "Download Complete!"
    mv "${MODEL_TORRENT_BASENAME%.*}" $MODEL_NAME
    fi
}

redownload () {
	echo "Deleting $MODEL_DIRECTORY"
	rm -rf ${MODEL_DIRECTORY}
	download_torrent
}

if [[ -d "${MODEL_DIRECTORY}" ]]; then
  redownload
else
	download_torrent
fi
