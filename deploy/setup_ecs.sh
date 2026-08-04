#!/usr/bin/env bash
# 在 Ubuntu 服务器上部署 DocChat（轻量版，不含 CLIP）。
#
# 用法（在服务器上以 root 执行）：
#   export DASHSCOPE_API_KEY=sk-xxxx
#   bash setup_ecs.sh
#
# 做的事：
#   1. 装系统依赖与 Python 虚拟环境
#   2. 拉取代码，安装轻量依赖（不含 torch，约 300MB）
#   3. 写入 .env，构建索引
#   4. 注册 systemd 服务（开机自启、崩溃自动重启）
#   5. 配置 nginx 反向代理到 80 端口
set -euo pipefail

APP_DIR=${APP_DIR:-/opt/docchat}
REPO=${REPO:-https://github.com/EtheXReal/basiclaw-rag.git}
BRANCH=${BRANCH:-main}
SERVICE=docchat
PORT=7860

if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
    echo "错误：请先设置 DASHSCOPE_API_KEY 环境变量再运行本脚本。"
    exit 1
fi

echo "==> [1/5] 安装系统依赖"
# 无人值守安装。缺这两个变量时 Ubuntu 22.04 会弹出交互式对话框卡住脚本：
#   - DEBIAN_FRONTEND=noninteractive 禁掉 debconf 的配置问答
#   - NEEDRESTART_MODE=a 让 needrestart 自动重启受影响的服务，
#     而不是弹出「Daemons using outdated libraries」让人选
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
export NEEDRESTART_SUSPEND=1
apt-get update -qq
apt-get install -y -qq python3-venv python3-pip git nginx

echo "==> [2/5] 拉取代码到 $APP_DIR"
if [ -d "$APP_DIR/.git" ]; then
    git -C "$APP_DIR" fetch --quiet origin
    git -C "$APP_DIR" checkout --quiet "$BRANCH"
    git -C "$APP_DIR" reset --hard --quiet "origin/$BRANCH"
else
    git clone --quiet --branch "$BRANCH" "$REPO" "$APP_DIR"
fi
cd "$APP_DIR"

echo "==> [3/5] 创建虚拟环境并安装依赖（轻量版，不含 torch）"
python3 -m venv .venv
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet -r deploy/requirements.lite.txt

echo "==> [4/5] 写入配置并构建索引"
cat > .env <<EOF
DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY}

# 关闭 CLIP：本机内存有限，且图像检索依赖 torch（约 1.5GB 常驻内存）
CLIP_ENABLED=false

# 无 Redis 服务，元数据走 SQLite
METADATA_BACKEND=sqlite
METADATA_PREFIX=basiclaw

# 索引已含文字层，无需 OCR
OCR_ENABLED=false

RERANK_ENABLED=true
RERANK_MODEL=gte-rerank-v2
RERANK_CANDIDATES=20

LLM_ENABLED=true
LLM_MODEL=qwen-turbo

# 绑 0.0.0.0 供 nginx 反代；对外只暴露 nginx 的 80 端口
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=${PORT}
EOF
chmod 600 .env

# 索引在服务器上现场构建：仅文本嵌入，约 600 次调用、一分钟量级。
# 比从本地 scp 上来更省事，且保证索引与当前代码版本一致。
./.venv/bin/python main.py --rebuild

echo "==> [5/5] 注册 systemd 服务与 nginx 反代"
cat > /etc/systemd/system/${SERVICE}.service <<EOF
[Unit]
Description=DocChat RAG service
After=network.target

[Service]
Type=simple
WorkingDirectory=${APP_DIR}
ExecStart=${APP_DIR}/.venv/bin/python app_gradio.py
Restart=always
RestartSec=5
# 内存兜底：超过后由 systemd 重启而非把整机拖垮
MemoryMax=900M

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/nginx/sites-available/${SERVICE} <<EOF
server {
    listen 80;
    server_name docchat.xreal.cc;

    location / {
        proxy_pass http://127.0.0.1:${PORT};
        proxy_http_version 1.1;
        # Gradio 依赖 WebSocket 推送结果，缺这两行会让检索永远转圈
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        # LLM 生成可能超过默认 60s
        proxy_read_timeout 300s;
    }
}
EOF
ln -sf /etc/nginx/sites-available/${SERVICE} /etc/nginx/sites-enabled/${SERVICE}
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

systemctl daemon-reload
systemctl enable --now ${SERVICE}

echo
echo "✅ 部署完成"
echo "   本机自检：curl -s -o /dev/null -w '%{http_code}\\n' http://127.0.0.1:${PORT}/"
echo "   查看日志：journalctl -u ${SERVICE} -f"
echo
echo "还需手动完成："
echo "   1. 阿里云安全组放行 80 端口"
echo "   2. Cloudflare 添加 A 记录 docchat -> 本机公网 IP，开启橙云（自动 HTTPS）"
