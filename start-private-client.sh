#!/bin/sh
sudo docker container stop private-client
sudo docker container remove private-client
sudo docker create -p 8282:8282 --ipc host --gpus 1 --volume petals-cache3:/root/.cache --name private-client hive-chat-private:latest
sudo docker container start private-client
