docker run \
 -it \
 --rm \
 --gpus all \
 -u $(id -u):$(id -g) \
 -v $PWD:/workspace \
 tf-autoaugment
