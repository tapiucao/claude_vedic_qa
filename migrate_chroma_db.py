"""
Script to migrate Chroma database to a new format or fix corrupted databases.
"""
import os
import sys
import shutil
import logging
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("db_migration")

def backup_db(db_dir):
    """Create a backup of the database directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"{db_dir}_backup_{timestamp}"
    
    try:
        # Copy the entire directory
        shutil.copytree(db_dir, backup_dir)
        logger.info(f"Created backup at: {backup_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to create backup: {str(e)}")
        return False

def fix_chroma_config(db_dir):
    """Try to fix corrupted Chroma config files."""
    # Look for chroma.sqlite3 file
    sqlite_file = os.path.join(db_dir, "chroma.sqlite3")
    if not os.path.exists(sqlite_file):
        logger.error(f"Database file not found: {sqlite_file}")
        return False
    
    # Check if there are any collection config files
    collections_dir = os.path.join(db_dir, "collections")
    if not os.path.exists(collections_dir):
        logger.warning("No collections directory found. Creating it.")
        os.makedirs(collections_dir, exist_ok=True)
    
    # Scan for collection configs
    fixed = False
    for root, dirs, files in os.walk(collections_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                logger.info(f"Examining config file: {file_path}")
                
                try:
                    # Read the config file
                    with open(file_path, 'r') as f:
                        config = json.load(f)
                    
                    # Check if '_type' field is missing
                    if '_type' not in config:
                        logger.info(f"Adding missing '_type' field to {file_path}")
                        config['_type'] = 'ChromaCollectionConfiguration'
                        
                        # Write back the fixed config
                        with open(file_path, 'w') as f:
                            json.dump(config, f, indent=2)
                        
                        fixed = True
                        logger.info(f"Fixed config file: {file_path}")
                except json.JSONDecodeError:
                    logger.error(f"Corrupted JSON in {file_path}. Cannot fix automatically.")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
    
    return fixed

def reset_db(db_dir):
    """Completely reset the database directory."""
    try:
        # Remove all files in the directory
        for item in os.listdir(db_dir):
            item_path = os.path.join(db_dir, item)
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        
        logger.info(f"Reset database directory: {db_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to reset database: {str(e)}")
        return False

def main():
    """Main function to run the migration."""
    if len(sys.argv) < 2:
        print("Usage: python migrate_chroma_db.py <db_directory> [--reset]")
        sys.exit(1)
    
    db_dir = sys.argv[1]
    reset_mode = "--reset" in sys.argv
    
    logger.info(f"Starting migration for database: {db_dir}")
    
    # Validate the directory
    if not os.path.isdir(db_dir):
        logger.error(f"Not a valid directory: {db_dir}")
        sys.exit(1)
    
    # Create backup
    if not backup_db(db_dir):
        logger.error("Failed to create backup. Aborting.")
        sys.exit(1)
    
    # Process the database
    if reset_mode:
        logger.info("Reset mode enabled. Database will be emptied.")
        if reset_db(db_dir):
            logger.info("Database reset successfully.")
            logger.info("You'll need to reload your documents with 'python app.py load'")
        else:
            logger.error("Failed to reset database.")
    else:
        logger.info("Attempting to fix database configuration...")
        if fix_chroma_config(db_dir):
            logger.info("Fixed database configuration successfully!")
        else:
            logger.info("No changes were made to the database.")
            logger.info("Consider using --reset if problems persist.")
    
    logger.info("Migration completed.")

if __name__ == "__main__":
    main()