# Updated: src/utils/exporter.py
"""
Export utilities for Vedic Knowledge AI.
Handles exporting data in various formats.
"""
import os
import json
import csv
import datetime
import logging
from typing import List, Dict, Any, Optional
import pandas as pd

from ..config import QA_LOGS_DIR, REPORTS_DIR, SUMMARIES_DIR

# Configure logging
logger = logging.getLogger(__name__)

class DataExporter:
    """Utility for exporting data to files."""

    @staticmethod
    def export_qa_log(question: str, answer: str, sources: Optional[List[str]] = None) -> str:
        """Export a question-answer pair to a log file."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"qa_log_{timestamp}.json"
        filepath = os.path.join(QA_LOGS_DIR, filename)

        data: Dict[str, Any] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "sources": sources or []
        }

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Exported QA log to {filepath}")
            return filepath
        except IOError as e:
            logger.error(f"Failed to write QA log to {filepath}: {e}")
            raise

    @staticmethod
    def export_text_summary(text_id: str, summary: str, metadata: Dict[str, Any]) -> str:
        """Export a text summary to a file."""
        filename = f"summary_{text_id}.md"
        filepath = os.path.join(SUMMARIES_DIR, filename)

        # Create markdown content
        content = f"# Summary of {metadata.get('title', text_id)}\n\n"
        content += f"- Source: {metadata.get('source', 'Unknown')}\n"
        content += f"- Date: {datetime.datetime.now().strftime('%Y-%m-%d')}\n\n"
        content += f"## Summary\n\n{summary}\n\n"

        # Add metadata section
        content += "## Metadata\n\n"
        for key, value in metadata.items():
            if key not in ['title', 'source'] and value is not None:
                if isinstance(value, (dict, list)):
                    content += f"- **{key}**: {json.dumps(value, ensure_ascii=False)}\n"
                else:
                    content += f"- **{key}**: {value}\n"

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Exported text summary to {filepath}")
            return filepath
        except IOError as e:
            logger.error(f"Failed to write text summary to {filepath}: {e}")
            raise

    @staticmethod
    def export_qa_batch(qa_pairs: List[Dict[str, Any]], title: str = "QA Batch") -> str:
        """Export a batch of Q&A pairs to both JSON and markdown."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        json_filename = f"qa_batch_{timestamp}.json"
        md_filename = f"qa_batch_{timestamp}.md"

        json_filepath = os.path.join(QA_LOGS_DIR, json_filename)
        md_filepath = os.path.join(QA_LOGS_DIR, md_filename)

        # Add timestamp
        data: Dict[str, Any] = {
            "title": title,
            "timestamp": datetime.datetime.now().isoformat(),
            "qa_pairs": qa_pairs
        }

        try:
            # Export to JSON
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Export to markdown
            md_content = f"# {title}\n\n"
            md_content += f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

            for i, qa in enumerate(qa_pairs):
                md_content += f"## Q{i+1}: {qa.get('question', 'No question')}\n\n"
                md_content += f"{qa.get('answer', 'No answer')}\n\n"

                if qa.get('sources'):
                    md_content += "**Sources:**\n\n"
                    # Ensure sources is a list before iterating
                    sources_list = qa['sources'] if isinstance(qa['sources'], list) else []
                    for j, source in enumerate(sources_list):
                        md_content += f"{j+1}. {source}\n"
                    md_content += "\n"

                md_content += "---\n\n"

            with open(md_filepath, 'w', encoding='utf-8') as f:
                f.write(md_content)

            logger.info(f"Exported QA batch to {json_filepath} and {md_filepath}")
            return json_filepath
        except IOError as e:
            logger.error(f"Failed to write QA batch files: {e}")
            raise
        except TypeError as e:
            logger.error(f"Type error during QA batch export (check sources format?): {e}")
            raise


    @staticmethod
    def export_report(report_data: Dict[str, Any], report_type: str) -> str:
        """Export a report in markdown format."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{report_type}_{timestamp}.md"
        filepath = os.path.join(REPORTS_DIR, filename)

        # Get report title and description
        title = report_data.get('title', f"{report_type.replace('_', ' ').title()} Report")
        description = report_data.get('description', '')

        # Create markdown content
        content = f"# {title}\n\n"
        content += f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        if description:
            content += f"{description}\n\n"

        # Add sections
        for section_name, section_data in report_data.get('sections', {}).items():
            content += f"## {section_name.replace('_', ' ').title()}\n\n"

            # Handle different types of section data
            if isinstance(section_data, str):
                content += f"{section_data}\n\n"
            elif isinstance(section_data, list):
                for item in section_data:
                    if isinstance(item, str):
                        content += f"- {item}\n"
                    elif isinstance(item, dict):
                        item_text = item.get('text', '')
                        item_details = item.get('details', '')
                        content += f"- **{item_text}**: {item_details}\n"
                    else: # Handle other potential list item types gracefully
                         content += f"- {str(item)}\n"
                content += "\n"
            elif isinstance(section_data, dict):
                for key, value in section_data.items():
                    content += f"### {key.replace('_', ' ').title()}\n\n"
                    content += f"{value}\n\n"
            else: # Handle other potential section data types gracefully
                 content += f"{str(section_data)}\n\n"

        try:
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Exported {report_type} report to {filepath}")
            return filepath
        except IOError as e:
            logger.error(f"Failed to write report to {filepath}: {e}")
            raise

    @staticmethod
    def export_statistics(stats: Dict[str, Any], name: str = "system_statistics") -> str:
        """Export statistics data in multiple formats."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        json_filename = f"{name}_{timestamp}.json"
        csv_filename = f"{name}_{timestamp}.csv"
        md_filename = f"{name}_{timestamp}.md"

        json_filepath = os.path.join(REPORTS_DIR, json_filename)
        csv_filepath = os.path.join(REPORTS_DIR, csv_filename)
        md_filepath = os.path.join(REPORTS_DIR, md_filename)

        try:
            # Export to JSON
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

            # Convert to DataFrame for CSV export
            flattened_data = DataExporter._flatten_dict(stats)
            df = pd.DataFrame([flattened_data])
            df.to_csv(csv_filepath, index=False)

            # Create markdown
            md_content = f"# {name.replace('_', ' ').title()} - {datetime.datetime.now().strftime('%Y-%m-%d')}\n\n"

            # Add sections based on top-level keys
            for section, data in stats.items():
                md_content += f"## {section.replace('_', ' ').title()}\n\n"

                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, dict):
                            md_content += f"### {key.replace('_', ' ').title()}\n\n"
                            for subkey, subvalue in value.items():
                                md_content += f"- **{subkey.replace('_', ' ').title()}**: {subvalue}\n"
                            md_content += "\n"
                        elif isinstance(value, list): # Handle lists within dicts
                             md_content += f"- **{key.replace('_', ' ').title()}**: {', '.join(map(str, value))}\n" # Simple comma separation
                        else:
                            md_content += f"- **{key.replace('_', ' ').title()}**: {value}\n"
                    md_content += "\n"
                elif isinstance(data, list):
                    for item in data:
                        md_content += f"- {str(item)}\n" # Ensure items are strings
                    md_content += "\n"
                else:
                    md_content += f"{str(data)}\n\n" # Ensure data is a string

            with open(md_filepath, 'w', encoding='utf-8') as f:
                f.write(md_content)

            logger.info(f"Exported statistics to {json_filepath}, {csv_filepath}, and {md_filepath}")
            return json_filepath
        except (IOError, pd.errors.EmptyDataError, TypeError) as e:
            logger.error(f"Failed to export statistics: {e}")
            raise


    @staticmethod
    def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten a nested dictionary for CSV export."""
        items: List[Tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(DataExporter._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Serialize list to JSON string for CSV compatibility
                try:
                    items.append((new_key, json.dumps(v)))
                except TypeError:
                     items.append((new_key, json.dumps([str(item) for item in v]))) # Fallback for non-serializable items
            else:
                items.append((new_key, v))

        return dict(items)

    @staticmethod
    def export_sanskrit_terms(terms_data: Dict[str, Dict[str, Any]]) -> str:
        """Export Sanskrit terms dictionary to a searchable markdown file."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"sanskrit_terms_{timestamp}.md"
        filepath = os.path.join(SUMMARIES_DIR, filename)

        content = "# Sanskrit Terms Dictionary\n\n"
        content += f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"


        # Sort terms alphabetically
        sorted_terms = sorted(terms_data.keys())

        if not sorted_terms:
             content += "No terms found.\n"
        else:
            content += "## Terms\n\n"
            # Add table of contents
            content += "### Table of Contents\n\n"
            for term in sorted_terms:
                anchor = term.lower().replace(' ', '-')
                content += f"- [{term}](#{anchor})\n"
            content += "\n---\n\n"

            # Add each term
            for term in sorted_terms:
                term_data = terms_data[term]
                anchor = term.lower().replace(' ', '-')
                content += f"### {term}\n\n" # Removed anchor from header, keep it for TOC links

                # Add devanagari if available
                if term_data.get('devanagari'):
                    content += f"**Devanagari:** {term_data['devanagari']}\n\n"

                # Add transliteration if available and different from term
                if term_data.get('transliteration') and term_data['transliteration'] != term:
                    content += f"**Transliteration:** {term_data['transliteration']}\n\n"

                # Add definition
                if term_data.get('definition'):
                    content += f"**Definition:** {term_data['definition']}\n\n"

                # Add etymology if available
                if term_data.get('etymology'):
                    content += f"**Etymology:** {term_data['etymology']}\n\n"

                # Add usage examples
                if term_data.get('examples'):
                    content += "**Examples:**\n\n"
                    for example in term_data['examples']:
                        content += f"- {example}\n"
                    content += "\n"

                # Add sources
                if term_data.get('sources'):
                    content += "**Sources:**\n\n"
                    for source in term_data['sources']:
                        content += f"- {source}\n"
                    content += "\n"

                # Add related terms
                if term_data.get('related_terms'):
                    content += "**Related Terms:**\n\n"
                    for related in term_data['related_terms']:
                        content += f"- {related}\n"
                    content += "\n"

                content += "---\n\n"

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Exported Sanskrit terms dictionary to {filepath}")
            return filepath
        except IOError as e:
            logger.error(f"Failed to write Sanskrit terms dictionary to {filepath}: {e}")
            raise