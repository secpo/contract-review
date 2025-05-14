#!/bin/bash
# 运行合同审查系统的启动脚本

# 检查.env文件是否存在
if [ ! -f .env ]; then
    echo "未找到.env文件，将从.env.example创建"
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "已从.env.example创建.env文件，请编辑该文件设置您的API密钥和其他配置"
        exit 1
    else
        echo "错误：.env.example文件也不存在，无法创建.env文件"
        exit 1
    fi
fi

# 加载.env文件中的环境变量
export $(grep -v '^#' .env | xargs)

# 检查关键环境变量
if [ "$AI_SERVICE_TYPE" = "openai" ]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "错误：使用OpenAI服务类型时，OPENAI_API_KEY必须设置"
        echo "请在.env文件中设置OPENAI_API_KEY"
        exit 1
    fi
    
    if [ -z "$OPENAI_BASE_URL" ]; then
        echo "警告：OPENAI_BASE_URL未设置，将使用默认值https://api.openai.com/v1"
        echo "如果您使用的是第三方兼容服务，请确保在.env文件中设置了正确的OPENAI_BASE_URL"
    fi
elif [ "$AI_SERVICE_TYPE" = "gemini" ]; then
    if [ -z "$GEMINI_API_KEY" ]; then
        echo "错误：使用Gemini服务类型时，GEMINI_API_KEY必须设置"
        echo "请在.env文件中设置GEMINI_API_KEY"
        exit 1
    fi
else
    echo "警告：未知的AI_SERVICE_TYPE: $AI_SERVICE_TYPE"
    echo "将使用默认值'openai'"
    export AI_SERVICE_TYPE=openai
fi

# 显示当前配置
echo "当前配置:"
echo "  - AI_SERVICE_TYPE: $AI_SERVICE_TYPE"
if [ "$AI_SERVICE_TYPE" = "openai" ]; then
    echo "  - OPENAI_BASE_URL: $OPENAI_BASE_URL"
    echo "  - OPENAI_MODEL_ID: $OPENAI_MODEL_ID"
    echo "  - OPENAI_API_KEY: ${OPENAI_API_KEY:0:5}... (已隐藏部分)"
elif [ "$AI_SERVICE_TYPE" = "gemini" ]; then
    echo "  - GEMINI_MODEL_ID: $GEMINI_MODEL_ID"
    echo "  - GEMINI_API_KEY: ${GEMINI_API_KEY:0:5}... (已隐藏部分)"
fi

# 运行程序
echo "正在启动程序..."
python3 convert-document-to-json.py

# 脚本结束
echo "程序执行完成"
