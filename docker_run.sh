# Test from Docker version 18.06.2-ce, build 6d37f41
docker run --runtime=nvidia -d -p 8889:8889 -v ${HOME}/experiments/photi-vision-data:/workspace/photi-vision-data photi-vision/parking-lot-detection:0.0.1
