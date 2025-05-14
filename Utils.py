import base64
import re
import json
import os
import mimetypes
import docx2txt
import PyPDF2

def open_as_bytes(file_path: str):
    """以二进制方式打开文件并返回base64编码"""
    with open(file_path, 'rb') as file:
        file_bytes = file.read()
        file_base64 = base64.b64encode(file_bytes)
    return file_base64

def read_text_file(file_path):
    """读取文本文件内容"""
    # Open the file in read mode
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
    return file_content

def extract_text_from_document(file_path):
    """从不同类型的文档中提取文本内容"""
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.docx':
        return extract_text_from_docx(file_path)
    elif file_extension == '.doc':
        # 注意：处理.doc文件可能需要额外的库
        return "DOC文件格式暂不支持直接提取文本，将使用AI服务处理"
    elif file_extension == '.txt':
        return read_text_file(file_path)
    else:
        return f"不支持的文件格式: {file_extension}"

def extract_text_from_pdf(pdf_path):
    """从PDF文件中提取文本"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
    except Exception as e:
        text = f"PDF提取错误: {str(e)}"
    return text

def extract_text_from_docx(docx_path):
    """从DOCX文件中提取文本"""
    try:
        text = docx2txt.process(docx_path)
        return text
    except Exception as e:
        return f"DOCX提取错误: {str(e)}"

def extract_json_from_string(input_string):
    try:
        # Parse the JSON string into a Python object (dictionary)
        if input_string.startswith('```json'):
            input_string = re.sub(r'^```json\s*|\s*```$', '', input_string, flags=re.DOTALL)

        json_object = json.loads(input_string)
        return json_object
    except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return None
    else:
        print("No valid JSON block found.")
        return None

def save_json_string_to_file(json_string, file_path):
    # Open the file in write mode and save the JSON string
    with open(file_path, 'w') as file:
        file.write(json_string)
