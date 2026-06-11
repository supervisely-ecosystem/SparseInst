VER="1.1.1"
TAG="supervisely/sparseinst:$VER"

docker build -t $TAG . && \
docker push $TAG