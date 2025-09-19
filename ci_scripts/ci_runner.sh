#!/bin/bash
# Usage: ./ci_runner.sh <commit_id> <artifact_dir> <ascend_id>

source /etc/environment

COMMIT_ID=$1
ARTIFACT_DIR=$2
ARTIFACT_DIR="${ARTIFACT_DIR}/${COMMIT_ID:0:7}_$(date --utc +%Y%m%d_%H%M%SZ)"
mkdir -p $ARTIFACT_DIR

if [ -n "$3" ]; then
  ASCEND_ID=$3
else
  ASCEND_ID=6
fi

IMAGE="cann_8.3.rc1.alpha002"
CONTAINER_NAME="ci_run_${COMMIT_ID}_$RANDOM"

cleanup() {
  echo "Caught SIGINT, killing container..."
  docker rm -f $CONTAINER_NAME
  exit 130
}

echo "Running CI for commit $COMMIT_ID in Docker..."

docker run --ipc=host --privileged --net=host \
 --device=/dev/davinci${ASCEND_ID}  --device=/dev/davinci_manager \
 --device=/dev/devmm_svm --device=/dev/hisi_hdc  \
 -e http_proxy="$http_proxy" -e https_proxy="$https_proxy" -e no_proxy="$no_proxy" \
 -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
 -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
 -v $ARTIFACT_DIR:/artifacts --name "$CONTAINER_NAME" -d "$IMAGE" tail -f /dev/null

trap cleanup SIGINT

echo "Copying scripts to docker container"
docker cp "$(dirname "$0")/ci_build_job.sh" "$CONTAINER_NAME":/root/ci_build_job.sh
docker cp "$(dirname "$0")/ci_run_tests.sh" "$CONTAINER_NAME":/root/ci_run_tests.sh
docker cp "$(dirname "$0")/run_test.sh" "$CONTAINER_NAME":/root/run_test.sh
docker cp "$(dirname "$0")/tests/." "$CONTAINER_NAME":/root/tests/


echo "Building ascend triton"
# Run the build job inside container (pass commit_id)
docker exec "$CONTAINER_NAME" bash -lc "/root/ci_build_job.sh '$COMMIT_ID'" > "$ARTIFACT_DIR/build_job.log" 2>&1
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Build failed with ${EXIT_CODE}"
    exit 1
fi

docker exec "$CONTAINER_NAME" bash -c "cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg" | grep toolkit_installed_version | awk -F'[:\\]]' '{print $2}' > "$ARTIFACT_DIR/cann_version.log"

echo "Running tests"
docker exec "$CONTAINER_NAME" bash -lc "/root/ci_run_tests.sh" > "$ARTIFACT_DIR/test_job.log" 2>&1

docker stop "$CONTAINER_NAME" > /dev/null
docker rm "$CONTAINER_NAME" > /dev/null

echo "CI run finished"
