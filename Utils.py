import base64
import re
import json

def open_as_bytes(pdf_filename:str):
    with open(pdf_filename, 'rb') as pdf_file:
        pdf_bytes = pdf_file.read()
        pdf_base64 = base64.b64encode(pdf_bytes)
    return pdf_base64

def read_text_file(file_path):
    # Open the file in read mode with GBK encoding
    try:
        with open(file_path, 'r', encoding='gbk') as file:
            file_content = file.read()
        return file_content
    except UnicodeDecodeError:
        # If GBK fails, try UTF-8
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
            return file_content
        except UnicodeDecodeError:
            # If both fail, use binary mode and decode with errors='ignore'
            with open(file_path, 'rb') as file:
                binary_content = file.read()
                return binary_content.decode('utf-8', errors='ignore')

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
    # Open the file in write mode with GBK encoding and save the JSON string
    try:
        with open(file_path, 'w', encoding='gbk') as file:
            file.write(json_string)
    except UnicodeEncodeError:
        # If GBK encoding fails, use UTF-8
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(json_string)
