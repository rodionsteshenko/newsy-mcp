#!/usr/bin/env python3
"""
User Preferences Migration Script

This script parses the user_preferences.txt file and migrates the preferences
to the database table, ensuring they are properly stored and retrievable.
"""

import sqlite3
import re
import sys
from typing import Dict, List
from rich.console import Console
from db_init import get_db_connection

# Configure console
console = Console()

# Database path
DB_PATH = "newsy.db"

def parse_user_preferences_file(file_path: str) -> Dict[str, List[str]]:
    """
    Parse the user preferences file into a dictionary of categories and items.
    
    Args:
        file_path: Path to the user preferences text file
        
    Returns:
        Dict with keys: 'topics_liked', 'topics_to_avoid', 'content_preferences'
    """
    preferences = {
        "topics_liked": [],
        "topics_to_avoid": [],
        "content_preferences": []
    }
    
    current_section = None
    console.print(f"Reading from {file_path}...")
    
    # Debug: Print each section as we process it
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            console.print(f"File has {len(lines)} lines")
            
            for i, line in enumerate(lines):
                line = line.strip()
                console.print(f"Line {i+1}: '{line}'"[:100] + ("..." if len(line) > 100 else ""))
                
                # Skip empty lines
                if not line:
                    continue
                
                # Detect section headers
                if "## Topics I Like" in line:
                    current_section = "topics_liked"
                    console.print(f"[bold green]Found Topics I Like section[/]")
                    continue
                elif "## Topics to Avoid" in line:
                    current_section = "topics_to_avoid"
                    console.print(f"[bold yellow]Found Topics to Avoid section[/]")
                    continue
                elif "## Content Preferences" in line:
                    current_section = "content_preferences"
                    console.print(f"[bold blue]Found Content Preferences section[/]")
                    continue
                
                # Process list items
                if line.startswith('- ') and current_section:
                    # Remove the '- ' prefix and any nested formatting
                    item = line[2:].strip()
                    console.print(f"  Adding to {current_section}: {item}")
                    
                    # Special case for nested lists (like TV Shows)
                    if ':' in item and not item.endswith(':'):
                        preferences[current_section].append(item)
                    elif not ':' in item:
                        preferences[current_section].append(item)
                    
                    # For nested items (indented with spaces)
                    if line.startswith('  - '):
                        # Get parent category from previous line
                        if i > 0 and ':' in lines[i-1]:
                            parent = lines[i-1].strip()[2:].split(':')[0].strip()
                            item = f"{parent} - {item}"
                            preferences[current_section].append(item)
                            console.print(f"  Added nested item: {item}")
                            
                        # Otherwise just add as is
                        else:
                            preferences[current_section].append(item)
    
    except Exception as e:
        console.print(f"[bold red]Error parsing preferences file: {e}[/]")
        return preferences
    
    return preferences

def save_preferences_to_database(preferences: Dict[str, List[str]], db_path: str = DB_PATH) -> bool:
    """
    Save the parsed preferences to the database.
    
    Args:
        preferences: Dictionary of preferences categories and items
        db_path: Path to the SQLite database
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            
            # Start a transaction
            conn.execute("BEGIN TRANSACTION")
            
            # Clear existing preferences
            cursor.execute("DELETE FROM user_preferences")
            
            # Insert topics liked
            for i, topic in enumerate(preferences["topics_liked"]):
                cursor.execute(
                    """
                    INSERT INTO user_preferences (key, value, category)
                    VALUES (?, ?, ?)
                    """,
                    (f"like_{i}", topic, "topic_liked")
                )
            
            # Insert topics to avoid
            for i, topic in enumerate(preferences["topics_to_avoid"]):
                cursor.execute(
                    """
                    INSERT INTO user_preferences (key, value, category)
                    VALUES (?, ?, ?)
                    """,
                    (f"avoid_{i}", topic, "topic_avoided")
                )
            
            # Insert content preferences
            for i, pref in enumerate(preferences["content_preferences"]):
                cursor.execute(
                    """
                    INSERT INTO user_preferences (key, value, category)
                    VALUES (?, ?, ?)
                    """,
                    (f"content_pref_{i}", pref, "content_preference")
                )
            
            # Commit the transaction
            conn.commit()
            
            # Verify the data was saved
            cursor.execute("SELECT COUNT(*) FROM user_preferences")
            count = cursor.fetchone()[0]
            
            console.print(f"[bold green]Successfully saved {count} preferences to database[/]")
            return True
            
    except Exception as e:
        console.print(f"[bold red]Error saving preferences to database: {e}[/]")
        return False

def verify_preferences_in_database(db_path: str = DB_PATH) -> bool:
    """
    Verify that preferences were correctly stored in the database.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        True if verification passed, False otherwise
    """
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            
            # Check each category
            categories = {
                "topic_liked": "topics liked",
                "topic_avoided": "topics to avoid",
                "content_preference": "content preferences"
            }
            
            for category, description in categories.items():
                cursor.execute(
                    "SELECT COUNT(*) FROM user_preferences WHERE category = ?",
                    (category,)
                )
                count = cursor.fetchone()[0]
                
                if count > 0:
                    console.print(f"[green]Found {count} {description} in database[/]")
                else:
                    console.print(f"[yellow]Warning: No {description} found in database[/]")
                    return False
            
            # Display some sample preferences
            for category, description in categories.items():
                cursor.execute(
                    "SELECT value FROM user_preferences WHERE category = ? LIMIT 5",
                    (category,)
                )
                items = [row[0] for row in cursor.fetchall()]
                console.print(f"[blue]Sample {description}:[/] {', '.join(items)}")
            
            return True
            
    except Exception as e:
        console.print(f"[bold red]Error verifying preferences: {e}[/]")
        return False

def main():
    """Main function to run the migration"""
    console.print("[bold blue]User Preferences Migration Tool[/]")
    
    # Parse preferences file
    prefs_file = "user_preferences.txt"
    console.print(f"Parsing preferences file: {prefs_file}")
    
    preferences = parse_user_preferences_file(prefs_file)
    
    # Display parsed preferences
    for category, items in preferences.items():
        console.print(f"[green]{category}:[/] {len(items)} items")
    
    # Save to database
    if save_preferences_to_database(preferences):
        console.print("[bold green]Successfully migrated preferences to database[/]")
    else:
        console.print("[bold red]Failed to migrate preferences to database[/]")
        return 1
    
    # Verify database
    console.print("\n[bold]Verifying preferences in database...[/]")
    if verify_preferences_in_database():
        console.print("[bold green]Verification passed![/]")
    else:
        console.print("[bold red]Verification failed![/]")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())