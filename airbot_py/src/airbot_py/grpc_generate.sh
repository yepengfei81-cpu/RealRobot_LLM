#!/bin/bash
python3 -m grpc_tools.protoc -Iairbot_grpc=../third_party/airbot_grpc/proto --python_out=. --pyi_out=. --grpc_python_out=. ../third_party/airbot_grpc/proto/common.proto ../third_party/airbot_grpc/proto/airbot_play.proto ../third_party/airbot_grpc/proto/mmk2.proto
