"""
File Parser Service
Save this file as: app/services/file_parser.py

This file reads different file types and converts them to text
"""

import pandas as pd
from pathlib import Path


class FileParser:
    """Parse different file types"""
    
    @staticmethod
    def parse_file(file_path: str):
        """
        Read file and return text content + dataframe (if applicable)
        
        Returns:
            tuple: (text_content, dataframe or None)
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        print(f"📄 Parsing {extension} file: {file_path.name}")
        
        # Check file type and parse accordingly
        if extension == '.md':
            text = FileParser._parse_markdown(file_path)
            return text, None
            
        elif extension in ['.xlsx', '.xls']:
            text, df = FileParser._parse_excel(file_path)
            return text, df
            
        elif extension == '.csv':
            text, df = FileParser._parse_csv(file_path)
            return text, df
            
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    
    @staticmethod
    def _parse_markdown(file_path: Path) -> str:
        """Read markdown file as text"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"✅ Parsed markdown: {len(content)} characters")
        return content
    
    
    @staticmethod
    def _parse_excel(file_path: Path):
        """
        Read Excel file and convert to text + dataframe
        Returns: (text, dataframe)
        """
        # Read Excel file
        df = pd.read_excel(file_path)
        
        # Create readable text version
        text_parts = []
        text_parts.append(f"# {file_path.name}\n")
        text_parts.append(f"Total Rows: {len(df)}\n")
        text_parts.append(f"Columns: {', '.join(df.columns)}\n\n")
        text_parts.append("## Data:\n")
        text_parts.append(df.to_string(index=False))
        
        text = '\n'.join(text_parts)
        
        print(f"✅ Parsed Excel: {len(df)} rows, {len(df.columns)} columns")
        return text, df
    
    
    @staticmethod
    def _parse_csv(file_path: Path):
        """
        Read CSV file and convert to text + dataframe
        Returns: (text, dataframe)
        """
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Create readable text version
        text_parts = []
        text_parts.append(f"# {file_path.name}\n")
        text_parts.append(f"Total Rows: {len(df)}\n")
        text_parts.append(f"Columns: {', '.join(df.columns)}\n\n")
        text_parts.append("## Data:\n")
        text_parts.append(df.to_string(index=False))
        
        text = '\n'.join(text_parts)
        
        print(f"✅ Parsed CSV: {len(df)} rows, {len(df.columns)} columns")
        return text, df
    
    
    @staticmethod
    def get_dataframe(file_path: str) -> pd.DataFrame:
        """
        Get pandas DataFrame from file (for SQL queries)
        Used when we need to query the data
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension == '.csv':
            return pd.read_csv(file_path)
        elif extension in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Cannot create DataFrame from {extension} files")