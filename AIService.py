import os
import json
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv
from DocumentProcessor import DocumentProcessor

load_dotenv(dotenv_path='.env', verbose=True)

class AIServiceBase(ABC):
    def __init__(self):
        # 初始化文档处理器
        self.document_processor = DocumentProcessor()

    @abstractmethod
    def process_document(self, document_path: str, prompt: str) -> str:
        pass

    @abstractmethod
    def get_embeddings(self, text: str) -> List[float]:
        pass

    def process_large_document(self, document_path: str, system_prompt: str, user_prompt: str) -> str:
        """
        处理大型文档，自动分块并合并结果

        Args:
            document_path: 文档路径
            system_prompt: 系统提示
            user_prompt: 用户提示

        Returns:
            处理结果
        """
        print(f"处理大型文档: {document_path}")

        # 使用DocumentProcessor分块处理文档
        try:
            chunks = self.document_processor.process_file(document_path)
            print(f"文档已分为 {len(chunks)} 个块")

            if len(chunks) == 1:
                # 如果只有一个块，直接处理
                return self.process_document(document_path, system_prompt, user_prompt)

            # 处理每个块
            results = []
            for i, chunk in enumerate(chunks):
                print(f"处理块 {i+1}/{len(chunks)}")

                # 创建临时文件
                temp_file_path = f"./data/temp/chunk_{i+1}.txt"
                os.makedirs("./data/temp", exist_ok=True)

                with open(temp_file_path, "w", encoding="utf-8") as f:
                    f.write(chunk)

                # 修改提示，指明这是文档的一部分
                chunk_user_prompt = f"{user_prompt}\n\n注意：这是文档的第 {i+1}/{len(chunks)} 部分。"

                # 处理块
                chunk_result = self.process_document(temp_file_path, system_prompt, chunk_user_prompt)
                results.append(chunk_result)

                # 删除临时文件
                try:
                    os.remove(temp_file_path)
                except:
                    pass

            # 如果有多个结果，合并它们
            if len(results) > 1:
                # 创建合并提示
                merge_prompt = f"我已经分析了一个文档的 {len(chunks)} 个部分，得到了以下结果：\n\n"
                for i, result in enumerate(results):
                    merge_prompt += f"--- 第 {i+1} 部分结果 ---\n{result}\n\n"

                merge_prompt += "请将这些结果合并为一个连贯的回答，确保信息不重复且保持原始格式。"

                # 创建临时文件
                temp_file_path = "./data/temp/merge_prompt.txt"
                with open(temp_file_path, "w", encoding="utf-8") as f:
                    f.write(merge_prompt)

                # 处理合并提示
                final_result = self.process_document(temp_file_path, system_prompt, "请合并以下分析结果：")

                # 删除临时文件
                try:
                    os.remove(temp_file_path)
                except:
                    pass

                return final_result
            else:
                return results[0]
        except Exception as e:
            print(f"处理大型文档时出错: {str(e)}")
            # 如果分块处理失败，尝试直接处理整个文档
            print("尝试直接处理整个文档")
            return self.process_document(document_path, system_prompt, user_prompt)

class OpenAIService(AIServiceBase):
    def __init__(self):
        super().__init__()
        from openai import OpenAI

        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            print("Warning: OPENAI_API_KEY not set. Using dummy_key.")
            print("If using third-party service, ensure OPENAI_BASE_URL is set.")
            self.api_key = "dummy_key"

        self.base_url = os.getenv('OPENAI_BASE_URL')
        if not self.base_url:
            print("Warning: OPENAI_BASE_URL not set. Using default.")
            self.base_url = 'https://api.openai.com/v1'

        self.model_id = os.getenv('OPENAI_MODEL_ID', 'gpt-4o')
        self.embedding_model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')

        print(f"Initializing OpenAI client:")
        print(f"  - API Base URL: {self.base_url}")
        print(f"  - Model ID: {self.model_id}")
        print(f"  - Embedding Model: {self.embedding_model}")

        try:
            # Initialize client with minimal parameters to avoid compatibility issues
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            print("OpenAI client initialized successfully")
        except TypeError as e:
            # If there's a TypeError, try with just the required parameters
            print(f"Error with standard initialization: {str(e)}")
            print("Trying alternative initialization...")
            try:
                import openai
                openai.api_key = self.api_key
                openai.base_url = self.base_url
                self.client = openai
                print("OpenAI client initialized using alternative method")
            except Exception as e2:
                print(f"Error with alternative initialization: {str(e2)}")
                raise
        except Exception as e:
            print(f"Error initializing OpenAI client: {str(e)}")
            raise

    def process_document(self, document_path: str, system_prompt: str, user_prompt: str) -> str:
        try:
            # Check if we're using the new client or the legacy client
            if hasattr(self.client, 'beta'):
                # New client (OpenAI class)
                return self._process_document_new_client(document_path, system_prompt, user_prompt)
            else:
                # Legacy client (openai module)
                return self._process_document_legacy_client(document_path, system_prompt, user_prompt)
        except Exception as e:
            print(f"Error in process_document: {str(e)}")
            # Try direct API call as fallback
            return self._process_document_direct_api(document_path, system_prompt, user_prompt)

    def _process_document_new_client(self, document_path: str, system_prompt: str, user_prompt: str) -> str:
        """Process document using the new OpenAI client"""
        from openai.types.beta.threads.message_create_params import (
            Attachment,
            AttachmentToolFileSearch,
        )

        print("Using new OpenAI client for document processing")

        try:
            assistant = self.client.beta.assistants.create(
                model=self.model_id,
                description="An assistant to extract information from documents.",
                tools=[{"type": "file_search"}],
                name="Document assistant",
                instructions=system_prompt,
            )

            thread = self.client.beta.threads.create()

            file = self.client.files.create(file=open(document_path, "rb"), purpose="assistants")

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

            run = self.client.beta.threads.runs.create_and_poll(
                thread_id=thread.id,
                assistant_id=assistant.id,
                timeout=1000
            )

            if run.status != "completed":
                raise Exception(f"Run failed: {run.status}")

            messages_cursor = self.client.beta.threads.messages.list(thread_id=thread.id)
            messages = [message for message in messages_cursor]

            return messages[0].content[0].text.value
        except Exception as e:
            print(f"Error in _process_document_new_client: {str(e)}")
            raise

    def _process_document_legacy_client(self, document_path: str, system_prompt: str, user_prompt: str) -> str:
        """Process document using the legacy OpenAI client"""
        import openai

        print("Using legacy OpenAI client for document processing")

        try:
            # For legacy client, we'll use chat completions with file content
            with open(document_path, 'rb') as file:
                file_content = file.read()

            import base64
            file_base64 = base64.b64encode(file_content).decode('utf-8')

            response = openai.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "file_content", "file_content": {
                            "mime_type": "application/octet-stream",
                            "data": file_base64
                        }}
                    ]}
                ],
                temperature=0.2
            )

            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in _process_document_legacy_client: {str(e)}")
            raise

    def _process_document_direct_api(self, document_path: str, system_prompt: str, user_prompt: str) -> str:
        """Process document using direct API calls as a last resort"""
        import requests
        import json
        import base64

        print("Using direct API calls for document processing")

        try:
            # Read file content
            with open(document_path, 'rb') as file:
                file_content = file.read()

            file_base64 = base64.b64encode(file_content).decode('utf-8')

            # Prepare request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "file_content", "file_content": {
                            "mime_type": "application/octet-stream",
                            "data": file_base64
                        }}
                    ]}
                ],
                "temperature": 0.2
            }

            # Make API call
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )

            if response.status_code != 200:
                print(f"API error: {response.status_code}")
                print(f"Response: {response.text}")
                raise Exception(f"API call failed with status {response.status_code}")

            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error in _process_document_direct_api: {str(e)}")
            raise

    def get_embeddings(self, text: str) -> List[float]:
        try:
            # Check if we're using the new client or the legacy client
            if hasattr(self.client, 'embeddings'):
                # New client (OpenAI class)
                return self._get_embeddings_new_client(text)
            else:
                # Legacy client (openai module)
                return self._get_embeddings_legacy_client(text)
        except Exception as e:
            print(f"Error in get_embeddings: {str(e)}")
            # Try direct API call as fallback
            return self._get_embeddings_direct_api(text)

    def _get_embeddings_new_client(self, text: str) -> List[float]:
        """Get embeddings using the new OpenAI client"""
        print("Using new OpenAI client for embeddings")
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error in _get_embeddings_new_client: {str(e)}")
            raise

    def _get_embeddings_legacy_client(self, text: str) -> List[float]:
        """Get embeddings using the legacy OpenAI client"""
        import openai

        print("Using legacy OpenAI client for embeddings")
        try:
            response = openai.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error in _get_embeddings_legacy_client: {str(e)}")
            raise

    def _get_embeddings_direct_api(self, text: str) -> List[float]:
        """Get embeddings using direct API calls as a last resort"""
        import requests
        import json

        print("Using direct API calls for embeddings")
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "model": self.embedding_model,
                "input": text
            }

            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=payload
            )

            if response.status_code != 200:
                print(f"API error: {response.status_code}")
                print(f"Response: {response.text}")
                raise Exception(f"API call failed with status {response.status_code}")

            response_data = response.json()
            return response_data["data"][0]["embedding"]
        except Exception as e:
            print(f"Error in _get_embeddings_direct_api: {str(e)}")
            # If all else fails, return a dummy embedding
            print("Returning dummy embedding")
            return [0.0] * 1536  # Standard OpenAI embedding size

class GeminiService(AIServiceBase):
    def __init__(self):
        super().__init__()
        import google.generativeai as genai

        self.api_key = os.getenv('GEMINI_API_KEY')
        self.model_id = os.getenv('GEMINI_MODEL_ID', 'gemini-1.5-pro')
        self.embedding_model = os.getenv('GEMINI_EMBEDDING_MODEL', 'embedding-001')

        print(f"Initializing Gemini client:")
        print(f"  - Model ID: {self.model_id}")
        print(f"  - Embedding Model: {self.embedding_model}")

        genai.configure(api_key=self.api_key)
        self.client = genai
        self.model = self.client.GenerativeModel(self.model_id)

    def process_document(self, document_path: str, system_prompt: str, user_prompt: str) -> str:
        import mimetypes

        mime_type, _ = mimetypes.guess_type(document_path)
        if not mime_type:
            mime_type = 'application/octet-stream'

        with open(document_path, 'rb') as f:
            file_content = f.read()

        combined_prompt = f"{system_prompt}\n\n{user_prompt}"

        chat = self.model.start_chat(history=[])

        response = chat.send_message(
            combined_prompt,
            generation_config={"temperature": 0.2},
            tools=[{"file_data": {"mime_type": mime_type, "data": file_content}}]
        )

        return response.text

    def get_embeddings(self, text: str) -> List[float]:
        print(f"Getting embeddings using Gemini model: {self.embedding_model}")
        try:
            embedding_model = self.client.get_model(self.embedding_model)
            result = embedding_model.embed_content(text)
            return result.embedding
        except Exception as e:
            print(f"Error getting embeddings with Gemini: {str(e)}")
            # If there's an error, try with the default model
            if self.embedding_model != 'embedding-001':
                print("Trying with default embedding model: embedding-001")
                try:
                    embedding_model = self.client.get_model('embedding-001')
                    result = embedding_model.embed_content(text)
                    return result.embedding
                except Exception as e2:
                    print(f"Error with default embedding model: {str(e2)}")

            # If all else fails, return a dummy embedding
            print("Returning dummy embedding")
            return [0.0] * 768  # Standard Gemini embedding size

def get_ai_service() -> AIServiceBase:
    load_dotenv(dotenv_path='.env', verbose=True)

    print("Current environment variables:")
    print(f"  - AI_SERVICE_TYPE: {os.getenv('AI_SERVICE_TYPE', 'not set')}")
    print(f"  - OPENAI_API_KEY: {'set' if os.getenv('OPENAI_API_KEY') else 'not set'}")
    print(f"  - OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL', 'not set')}")
    print(f"  - GEMINI_API_KEY: {'set' if os.getenv('GEMINI_API_KEY') else 'not set'}")

    service_type = os.getenv('AI_SERVICE_TYPE', 'openai').lower()
    print(f"Using AI service type: {service_type}")

    if service_type == 'openai':
        return OpenAIService()
    elif service_type == 'gemini':
        return GeminiService()
    else:
        raise ValueError(f"Unsupported AI service type: {service_type}")
