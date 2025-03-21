#!/bin/bash

# launch mlflow
mlflow server --host 0.0.0.0 --port 8080 &
MLFLOW_PID=$!

# launch tensorboard
tensorboard --host=0.0.0.0 --logdir=./runs --port=6006 &
TENSORBOARD_PID=$!

echo "MLFlow PID: $MLFLOW_PID"
echo "Tensorboard PID: $TENSORBOARD_PID"

cleanup(){
    echo "Shutting down logging services..."
    kill $MLFLOW_PID $TENSORBOARD_PID
    exit 0
}

trap cleanup SIGINT SIGTERM

while true; do
    sleep 1
done