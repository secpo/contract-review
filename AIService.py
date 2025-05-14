import os
import json
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env', verbose=True)

class AIServiceBase(ABC):
    @abstractmethod
    def process_document(self, document_path: str, prompt: str) -> str:
        pass

    @abstractmethod
    def get_embeddings(self, text: str) -> List[float]:
        pass

class OpenAIService(AIServiceBase):
    def __init__(self):
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
        import google.generativeai as genai

        self.api_key = os.getenv('GEMINI_API_KEY')
        self.model_id = os.getenv('GEMINI_MODEL_ID', 'gemini-1.5-pro')

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
        embedding_model = self.client.get_model('embedding-001')
        result = embedding_model.embed_content(text)
        return result.embedding

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
