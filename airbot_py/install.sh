#!/bin/bash

# usage: bash install.sh

set -x  # 移除 -e，允许命令失败后继续执行

SOURCE=https://pypi.mirrors.ustc.edu.cn/simple

# 假设 pip 已经安装，跳过 apt 命令
python3 -m pip install --upgrade pip -i ${SOURCE}
pip install setuptools==64 -i ${SOURCE}

pip install -e . --use-pep517 -i ${SOURCE}

pushd src/airbot_py
bash grpc_generate.sh
popd

pushd src/third_party/airbot_grpc
bash generate_python.sh
pip install -e . --use-pep517 -i ${SOURCE}
popd

pushd src/third_party/mmk2_types
pip install -e . --use-pep517 -i ${SOURCE}
popd

pip uninstall opencv-python-headless -y || true
pip install opencv-python -i ${SOURCE}