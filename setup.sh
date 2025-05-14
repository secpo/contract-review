#!/bin/bash
# 安装脚本 - 确保所有依赖项正确安装

echo "开始安装依赖项..."

# 检查Python版本
python_version=$(python3 --version 2>&1)
echo "检测到Python版本: $python_version"

# 创建虚拟环境（如果不存在）
if [ ! -d ".venv" ]; then
    echo "创建Python虚拟环境..."
    python3 -m venv .venv
    echo "虚拟环境创建完成"
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source .venv/bin/activate

# 安装基本依赖
echo "安装基本依赖项..."
pip install -r requirements.txt

# 确保httpx版本正确
current_httpx=$(pip show httpx | grep Version | awk '{print $2}')
echo "当前httpx版本: $current_httpx"

if [[ $(echo "$current_httpx" | awk -F. '{ print ($1 * 100 + $2) }') -ge 2800 ]]; then
    echo "httpx版本过高，降级到0.27.0..."
    pip install httpx==0.27.0
    new_httpx=$(pip show httpx | grep Version | awk '{print $2}')
    echo "httpx已降级到: $new_httpx"
else
    echo "httpx版本正确，无需降级"
fi

# 检查.env文件
if [ ! -f ".env" ]; then
    echo "未找到.env文件，从.env.example创建..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo ".env文件已创建，请编辑该文件设置您的API密钥和其他配置"
    else
        echo "创建示例.env文件..."
        cat > .env.example << 'EOF'
# 选择AI服务类型: openai 或 gemini
AI_SERVICE_TYPE=openai

# OpenAI配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL_ID=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Gemini配置
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL_ID=gemini-1.5-pro

# Neo4j配置
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
EOF
        cp .env.example .env
        echo ".env文件已创建，请编辑该文件设置您的API密钥和其他配置"
    fi
else
    echo ".env文件已存在"
fi

# 确保数据目录存在
mkdir -p ./data/input
mkdir -p ./data/output
mkdir -p ./data/debug

echo "安装完成！"
echo "请确保编辑.env文件设置您的API密钥和其他配置"
echo "您可以使用以下命令运行程序："
echo "  python convert-document-to-json.py - 处理文档"
echo "  python create_graph_from_json.py - 创建知识图谱"
echo "  python test_agent.py - 运行终端问答代理"
echo "  streamlit run app.py - 运行Web界面问答代理"
