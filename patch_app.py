"""
Quick patch script for app.py
"""
import os
import sys
import shutil

def create_backup(filename):
    """Create a backup of the file."""
    backup_filename = f"{filename}.bak"
    shutil.copy2(filename, backup_filename)
    print(f"Created backup: {backup_filename}")

def patch_file():
    """Patch the app.py file."""
    app_file = 'app.py'
    
    # Create backup
    create_backup(app_file)
    
    # Read the file
    with open(app_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the import for vector_store
    content = content.replace(
        "from src.knowledge_base.vector_store import VedicVectorStore",
        "from src.knowledge_base.vector_store_fixed import VedicVectorStore"
    )
    
    # Replace the SCRAPING_INTERVAL line in config.py if it exists
    config_file = 'src/config.py'
    if os.path.exists(config_file):
        create_backup(config_file)
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        # Replace the problematic line
        config_content = config_content.replace(
            'SCRAPING_INTERVAL = int(os.getenv("SCRAPING_INTERVAL", "86400"))  # Default to daily',
            'SCRAPING_INTERVAL = 86400  # Default to daily (24 hours in seconds)'
        )
        
        # Also update the DB_DIR path
        config_content = config_content.replace(
            'DB_DIR = os.getenv("DB_DIR", os.path.join(BASE_DIR, "data", "db"))',
            'DB_DIR = os.getenv("DB_DIR", os.path.join(BASE_DIR, "data", "db_new"))'
        )
        
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"Updated {config_file}")
    
    # Write the updated content back
    with open(app_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Updated {app_file}")
    print("Patch applied successfully!")

if __name__ == "__main__":
    patch_file()