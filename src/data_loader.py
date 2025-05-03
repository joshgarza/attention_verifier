import os

def load_document(file_path):
    """Loads text content from a file."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    try:
        # Start with basic text, add PDF/other handling later if needed
        if file_path.lower().endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        # Add elif clauses for .pdf, .json etc. here
        else:
            print(f"Error: Unsupported file type: {file_path}")
            return None
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None