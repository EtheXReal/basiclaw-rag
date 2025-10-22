#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_DIR=${VENV_DIR:-.venv}

echo "==> 创建虚拟环境 (${VENV_DIR})..."
if [ ! -d "${VENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "虚拟环境已存在，跳过创建。"
fi

ACTIVATE="${VENV_DIR}/bin/activate"
if [ ! -f "${ACTIVATE}" ]; then
  echo "未找到虚拟环境激活脚本：${ACTIVATE}"
  exit 1
fi

echo "==> 安装项目依赖..."
source "${ACTIVATE}"
pip install --upgrade pip
pip install -r requirements.txt

cat <<'INFO'

==> 安装完成！
请执行以下命令配置环境变量：

    source .venv/bin/activate
    export DASHSCOPE_API_KEY="你的DashScopeKey"
    export VECTOR_DATA_DIR="/path/to/faiss_data"   # 可选

Redis 默认连接 localhost:6379；如需更改可在 .env 中填写 REDIS_HOST/PORT。

INFO
