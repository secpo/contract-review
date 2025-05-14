import os
import re
import math
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
import tiktoken

# 支持的文件类型
class FileType:
    TXT = "txt"
    DOC = "doc"
    DOCX = "docx"
    PDF = "pdf"

class DocumentProcessor:
    """
    文档处理器类，用于读取和处理各种文本文件，并进行智能分块
    支持的文件格式：TXT, DOC, DOCX, PDF(文本型)
    """
    
    def __init__(self):
        """初始化文档处理器"""
        # 加载环境变量
        load_dotenv(dotenv_path='.env', verbose=True)
        
        # 获取MAX_TOKEN参数，默认为4000
        self.max_token = int(os.getenv('MAX_TOKEN', '4000'))
        print(f"文档处理器初始化完成，最大Token数: {self.max_token}")
        
        # 初始化tiktoken编码器，用于计算token数量
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")  # OpenAI的编码器
        except:
            try:
                self.encoding = tiktoken.encoding_for_model("gpt-4")
            except:
                print("警告: 无法加载tiktoken编码器，将使用近似计算方法")
                self.encoding = None
    
    def process_file(self, file_path: str) -> List[str]:
        """
        处理文件并返回分块后的内容列表
        
        Args:
            file_path: 文件路径
            
        Returns:
            分块后的内容列表
        """
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 获取文件类型
        file_type = self._get_file_type(file_path)
        print(f"处理文件: {file_path}, 类型: {file_type}")
        
        # 读取文件内容
        content = self._read_file(file_path, file_type)
        
        # 分块处理
        chunks = self._chunk_content(content)
        print(f"文件已分为 {len(chunks)} 个块")
        
        return chunks
    
    def _get_file_type(self, file_path: str) -> str:
        """获取文件类型"""
        _, ext = os.path.splitext(file_path)
        return ext.lower().replace('.', '')
    
    def _read_file(self, file_path: str, file_type: str) -> str:
        """根据文件类型读取文件内容"""
        if file_type == FileType.TXT:
            return self._read_txt_file(file_path)
        elif file_type == FileType.DOC:
            return self._read_doc_file(file_path)
        elif file_type == FileType.DOCX:
            return self._read_docx_file(file_path)
        elif file_type == FileType.PDF:
            return self._read_pdf_file(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_type}")
    
    def _read_txt_file(self, file_path: str) -> str:
        """读取TXT文件"""
        try:
            # 尝试使用不同编码读取文件
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    print(f"成功使用 {encoding} 编码读取文件")
                    return content
                except UnicodeDecodeError:
                    continue
            
            # 如果所有编码都失败，使用二进制模式读取并忽略错误
            with open(file_path, 'rb') as f:
                content = f.read().decode('utf-8', errors='ignore')
            print("使用二进制模式读取文件并忽略编码错误")
            return content
        except Exception as e:
            raise IOError(f"读取TXT文件时出错: {str(e)}")
    
    def _read_doc_file(self, file_path: str) -> str:
        """读取DOC文件"""
        try:
            # 尝试使用antiword
            try:
                import subprocess
                result = subprocess.run(['antiword', file_path], capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout
            except:
                pass
            
            # 尝试使用textract
            try:
                import textract
                text = textract.process(file_path).decode('utf-8')
                return text
            except:
                pass
            
            # 尝试使用pywin32 (仅Windows)
            try:
                import win32com.client
                word = win32com.client.Dispatch("Word.Application")
                word.Visible = False
                doc = word.Documents.Open(file_path)
                text = doc.Content.Text
                doc.Close()
                word.Quit()
                return text
            except:
                pass
            
            raise ImportError("无法读取DOC文件，请安装必要的依赖: antiword, textract 或 pywin32")
        except Exception as e:
            raise IOError(f"读取DOC文件时出错: {str(e)}")
    
    def _read_docx_file(self, file_path: str) -> str:
        """读取DOCX文件"""
        try:
            import docx2txt
            text = docx2txt.process(file_path)
            return text
        except ImportError:
            try:
                # 尝试安装docx2txt
                import subprocess
                subprocess.check_call(["pip", "install", "docx2txt"])
                import docx2txt
                text = docx2txt.process(file_path)
                return text
            except:
                pass
            
            try:
                # 尝试使用python-docx
                from docx import Document
                doc = Document(file_path)
                full_text = []
                for para in doc.paragraphs:
                    full_text.append(para.text)
                return '\n'.join(full_text)
            except ImportError:
                raise ImportError("无法读取DOCX文件，请安装必要的依赖: docx2txt 或 python-docx")
        except Exception as e:
            raise IOError(f"读取DOCX文件时出错: {str(e)}")
    
    def _read_pdf_file(self, file_path: str) -> str:
        """读取PDF文件"""
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:  # 确保页面有文本
                        text += page_text + "\n\n"  # 添加额外换行符分隔页面
            return text
        except ImportError:
            try:
                # 尝试安装PyPDF2
                import subprocess
                subprocess.check_call(["pip", "install", "PyPDF2"])
                import PyPDF2
                text = ""
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text += page.extract_text() + "\n\n"
                return text
            except:
                pass
            
            try:
                # 尝试使用pdfminer
                from pdfminer.high_level import extract_text
                text = extract_text(file_path)
                return text
            except ImportError:
                raise ImportError("无法读取PDF文件，请安装必要的依赖: PyPDF2 或 pdfminer.six")
        except Exception as e:
            raise IOError(f"读取PDF文件时出错: {str(e)}")
    
    def _count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        if self.encoding:
            # 使用tiktoken计算
            return len(self.encoding.encode(text))
        else:
            # 近似计算：英文约为每4个字符1个token，中文约为每1个字符1个token
            english_chars = len(re.findall(r'[a-zA-Z0-9]', text))
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            other_chars = len(text) - english_chars - chinese_chars
            return chinese_chars + math.ceil(english_chars / 4) + math.ceil(other_chars / 2)
    
    def _chunk_content(self, content: str) -> List[str]:
        """
        将内容按照自然段落分块，确保每块不超过max_token
        保留原始换行符和格式
        """
        # 按段落分割内容（保留空行）
        paragraphs = re.split(r'(\n+)', content)
        
        # 重组段落和换行符
        elements = []
        for i in range(0, len(paragraphs), 2):
            if i < len(paragraphs):
                para = paragraphs[i]
                # 添加段落
                elements.append(para)
                # 添加换行符（如果存在）
                if i + 1 < len(paragraphs):
                    elements.append(paragraphs[i + 1])
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for element in elements:
            # 计算当前元素的token数
            element_tokens = self._count_tokens(element)
            
            # 如果当前块加上新元素会超过max_token
            if current_tokens + element_tokens > self.max_token and current_chunk:
                # 保存当前块并开始新块
                chunks.append(current_chunk)
                current_chunk = element
                current_tokens = element_tokens
            else:
                # 将元素添加到当前块
                current_chunk += element
                current_tokens += element_tokens
        
        # 添加最后一个块（如果有）
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

# 示例用法
if __name__ == "__main__":
    processor = DocumentProcessor()
    
    # 示例：处理TXT文件
    try:
        chunks = processor.process_file("./data/input/sample.txt")
        print(f"共分为 {len(chunks)} 个块")
        for i, chunk in enumerate(chunks):
            print(f"块 {i+1} 的token数: {processor._count_tokens(chunk)}")
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
