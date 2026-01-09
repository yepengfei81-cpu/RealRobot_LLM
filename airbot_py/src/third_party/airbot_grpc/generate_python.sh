#!/bin/bash

python3 -m grpc_tools.protoc -Iairbot_grpc=../airbot_grpc --proto_path=airbot_grpc=./proto --python_out=. --grpc_python_out=. --pyi_out=. proto/common.proto proto/airbot_play.proto proto/mmk2.proto
