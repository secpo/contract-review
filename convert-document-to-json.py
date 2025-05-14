import os
import json
import sys
import locale
from Utils import read_text_file, save_json_string_to_file, extract_json_from_string
from AIService import get_ai_service

# Print system information for debugging
print(f"Python version: {sys.version}")
print(f"Default encoding: {sys.getdefaultencoding()}")
print(f"Filesystem encoding: {sys.getfilesystemencoding()}")
print(f"Locale: {locale.getdefaultlocale()}")
print(f"Current working directory: {os.getcwd()}")

# Load the system instruction and extraction prompt
try:
    system_instruction = read_text_file('./prompts/system_prompt.txt')
    extraction_prompt = read_text_file('./prompts/contract_extraction_prompt.txt')
    print("Successfully loaded prompts")
except Exception as e:
    print(f"Error loading prompts: {str(e)}")
    sys.exit(1)

# Get AI service instance
try:
    ai_service = get_ai_service()
    print("Successfully initialized AI service")
except Exception as e:
    print(f"Error initializing AI service: {str(e)}")
    sys.exit(1)

def process_document(document_path):
    """Process document and extract information"""
    print(f'Processing file: {document_path}')
    
    # Check if file exists
    if not os.path.exists(document_path):
        print(f"Error: File does not exist: {document_path}")
        return None
        
    # Check file size
    file_size = os.path.getsize(document_path)
    print(f"File size: {file_size} bytes")
    
    try:
        # Use AI service to process document
        complete_response = ai_service.process_document(
            document_path=document_path,
            system_prompt=system_instruction,
            user_prompt=extraction_prompt
        )
        return complete_response
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Supported file extensions
    supported_extensions = ['.pdf', '.docx', '.doc', '.txt']
    
    # Get all supported format files from input directory
    document_files = []
    try:
        for filename in os.listdir('./data/input/'):
            _, ext = os.path.splitext(filename)
            if ext.lower() in supported_extensions:
                document_files.append(filename)
    except Exception as e:
        print(f"Error reading input directory: {str(e)}")
        sys.exit(1)
    
    if not document_files:
        print("No supported document files found. Supported formats: PDF, DOCX, DOC, TXT")
        return
    
    # Ensure output directories exist
    os.makedirs('./data/debug/', exist_ok=True)
    os.makedirs('./data/output/', exist_ok=True)
    
    for document_filename in document_files:
        try:
            input_path = os.path.join('./data/input/', document_filename)
            
            # Process document
            complete_response = process_document(input_path)
            if not complete_response:
                print(f"Failed to process {document_filename}")
                continue
            
            # Save complete response for debugging
            debug_filename = os.path.join('./data/debug/', f'complete_response_{document_filename}.json')
            save_json_string_to_file(complete_response, debug_filename)
            
            # Try to parse response as JSON
            try:
                contract_json = extract_json_from_string(complete_response)
                if contract_json:
                    # Store as valid JSON
                    json_string = json.dumps(contract_json, indent=4, ensure_ascii=False)
                    output_filename = os.path.join('./data/output/', f'{document_filename}.json')
                    save_json_string_to_file(json_string, output_filename)
                    print(f'Successfully processed: {document_filename}')
                else:
                    print(f'Failed to extract JSON from response: {document_filename}')
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
        except Exception as e:
            print(f"Error processing file {document_filename}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
