VER="1.1.0"
TAG="supervisely/sparseinst:$VER"

docker build -t $TAG . && \
docker push $TAG