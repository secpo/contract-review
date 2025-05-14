import os
import json
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(dotenv_path='.env', verbose=True)

class AIServiceBase(ABC):
    """基础AI服务类，定义通用接口"""

    @abstractmethod
    def process_document(self, document_path: str, prompt: str) -> str:
        """处理文档并返回结果"""
        pass

    @abstractmethod
    def get_embeddings(self, text: str) -> List[float]:
        """获取文本的嵌入向量"""
        pass

class OpenAIService(AIServiceBase):
    """OpenAI服务实现"""

    def __init__(self):
        from openai import OpenAI

        # 从环境变量获取配置
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            # 如果API密钥未设置，使用一个默认值以满足OpenAI库的要求
            # 注意：这只是为了通过初始化，实际请求仍需要有效的API密钥
            print("警告: OPENAI_API_KEY环境变量未设置。使用默认值'dummy_key'。")
            print("如果您使用的是第三方兼容服务，请确保设置了正确的OPENAI_BASE_URL。")
            self.api_key = "dummy_key"  # 使用一个虚拟密钥

        self.base_url = os.getenv('OPENAI_BASE_URL')
        if not self.base_url:
            print("警告: OPENAI_BASE_URL环境变量未设置。使用默认值'https://api.openai.com/v1'。")
            print("如果您使用的是第三方兼容服务，请确保设置了正确的OPENAI_BASE_URL。")
            self.base_url = 'https://api.openai.com/v1'

        self.model_id = os.getenv('OPENAI_MODEL_ID', 'gpt-4o')
        self.embedding_model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')

        print(f"使用以下配置初始化OpenAI客户端:")
        print(f"  - API基础URL: {self.base_url}")
        print(f"  - 模型ID: {self.model_id}")
        print(f"  - 嵌入模型: {self.embedding_model}")

        # 初始化客户端
        try:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            print("OpenAI客户端初始化成功")
        except Exception as e:
            print(f"初始化OpenAI客户端时出错: {str(e)}")
            raise

    def process_document(self, document_path: str, system_prompt: str, user_prompt: str) -> str:
        """使用OpenAI处理文档"""
        from openai.types.beta.threads.message_create_params import (
            Attachment,
            AttachmentToolFileSearch,
        )

        # 创建助手
        assistant = self.client.beta.assistants.create(
            model=self.model_id,
            description="An assistant to extract information from documents.",
            tools=[{"type": "file_search"}],
            name="Document assistant",
            instructions=system_prompt,
        )

        # 创建线程
        thread = self.client.beta.threads.create()

        # 上传文件
        file = self.client.files.create(file=open(document_path, "rb"), purpose="assistants")

        # 创建消息
        self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            attachments=[
                Attachment(
                    file_id=file.id,
                    tools=[AttachmentToolFileSearch(type="file_search")]
                )
            ],
            content=user_prompt,
        )

        # 运行线程
        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant.id,
            timeout=1000
        )

        if run.status != "completed":
            raise Exception("Run failed:", run.status)

        # 获取消息
        messages_cursor = self.client.beta.threads.messages.list(thread_id=thread.id)
        messages = [message for message in messages_cursor]

        # 返回结果
        return messages[0].content[0].text.value

    def get_embeddings(self, text: str) -> List[float]:
        """获取文本的嵌入向量"""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding

class GeminiService(AIServiceBase):
    """Google Gemini服务实现"""

    def __init__(self):
        import google.generativeai as genai

        # 从环境变量获取配置
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.model_id = os.getenv('GEMINI_MODEL_ID', 'gemini-1.5-pro')

        # 初始化客户端
        genai.configure(api_key=self.api_key)
        self.client = genai
        self.model = self.client.GenerativeModel(self.model_id)

    def process_document(self, document_path: str, system_prompt: str, user_prompt: str) -> str:
        """使用Gemini处理文档"""
        import mimetypes

        # 确定文件类型
        mime_type, _ = mimetypes.guess_type(document_path)
        if not mime_type:
            mime_type = 'application/octet-stream'

        # 读取文件内容
        with open(document_path, 'rb') as f:
            file_content = f.read()

        # 创建多部分请求
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"

        # 创建聊天会话
        chat = self.model.start_chat(history=[])

        # 发送带有文件的消息
        response = chat.send_message(
            combined_prompt,
            generation_config={"temperature": 0.2},
            tools=[{"file_data": {"mime_type": mime_type, "data": file_content}}]
        )

        # 返回结果
        return response.text

    def get_embeddings(self, text: str) -> List[float]:
        """获取文本的嵌入向量"""
        embedding_model = self.client.get_model('embedding-001')
        result = embedding_model.embed_content(text)
        return result.embedding

def get_ai_service() -> AIServiceBase:
    """根据环境变量配置返回相应的AI服务实例"""
    # 确保环境变量已加载
    load_dotenv(dotenv_path='.env', verbose=True)

    # 打印当前环境变量状态
    print("当前环境变量状态:")
    print(f"  - AI_SERVICE_TYPE: {os.getenv('AI_SERVICE_TYPE', '未设置')}")
    print(f"  - OPENAI_API_KEY: {'已设置' if os.getenv('OPENAI_API_KEY') else '未设置'}")
    print(f"  - OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL', '未设置')}")
    print(f"  - GEMINI_API_KEY: {'已设置' if os.getenv('GEMINI_API_KEY') else '未设置'}")

    service_type = os.getenv('AI_SERVICE_TYPE', 'openai').lower()
    print(f"使用AI服务类型: {service_type}")

    if service_type == 'openai':
        return OpenAIService()
    elif service_type == 'gemini':
        return GeminiService()
    else:
        raise ValueError(f"不支持的AI服务类型: {service_type}")
