import os
from pathlib import Path
from docling.document_converter import DocumentConverter

def main():
    base_dir = Path("dataset/academic/research/pdf+gold")
    
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return
        
    print(f"Scanning directory: {base_dir}")
    pdf_files = list(base_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found.")
        return
        
    converter = DocumentConverter()
    
    for pdf_path in pdf_files:
        md_path = pdf_path.with_suffix(".md")
        if md_path.exists():
            print(f"Skipping {pdf_path.name}, markdown already exists.")
            continue
            
        print(f"Parsing {pdf_path.name}...")
        try:
            result = converter.convert(str(pdf_path))
            md_content = result.document.export_to_markdown()
            
            md_path.write_text(md_content, encoding="utf-8")
            print(f"Saved to {md_path.name}")
        except Exception as e:
            print(f"Error parsing {pdf_path.name}: {e}")

if __name__ == "__main__":
    main()
