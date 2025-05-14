import os
import json
from Utils import read_text_file, save_json_string_to_file, extract_json_from_string
from AIService import get_ai_service
import re

# 加载系统指令和提取提示
system_instruction = read_text_file('./prompts/system_prompt.txt')
extraction_prompt = read_text_file('./prompts/contract_extraction_prompt.txt')

# 获取AI服务实例
ai_service = get_ai_service()

def process_document(document_path):
    """处理文档并提取信息"""
    print(f'处理文件: {document_path}')
    
    # 使用AI服务处理文档
    complete_response = ai_service.process_document(
        document_path=document_path,
        system_prompt=system_instruction,
        user_prompt=extraction_prompt
    )
    
    # 返回提取的内容
    return complete_response

def main():
    # 支持的文件扩展名
    supported_extensions = ['.pdf', '.docx', '.doc', '.txt']
    
    # 获取输入目录中的所有支持格式的文件
    document_files = []
    for filename in os.listdir('./data/input/'):
        _, ext = os.path.splitext(filename)
        if ext.lower() in supported_extensions:
            document_files.append(filename)
    
    if not document_files:
        print("未找到支持的文档文件。支持的格式有：PDF, DOCX, DOC, TXT")
        return
    
    for document_filename in document_files:
        try:
            # 提取文档内容
            complete_response = process_document('./data/input/' + document_filename)
            
            # 保存完整响应用于调试
            debug_filename = f'./data/debug/complete_response_{document_filename}.json'
            save_json_string_to_file(complete_response, debug_filename)
            
            # 尝试将响应加载为有效的JSON
            try:
                contract_json = extract_json_from_string(complete_response)
                if contract_json:
                    # 存储为有效的JSON，以便稍后导入到KG
                    json_string = json.dumps(contract_json, indent=4)
                    output_filename = f'./data/output/{document_filename}.json'
                    save_json_string_to_file(json_string, output_filename)
                    print(f'成功处理: {document_filename}')
                else:
                    print(f'无法从响应中提取JSON: {document_filename}')
            except json.JSONDecodeError as e:
                print(f"JSON解码失败: {e}")
        except Exception as e:
            print(f"处理文件 {document_filename} 时出错: {str(e)}")

if __name__ == '__main__':
    main()
