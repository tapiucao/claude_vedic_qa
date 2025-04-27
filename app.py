"""
Main application for Vedic Knowledge AI.
Integrates all components and provides a command-line interface.
"""
import os
import sys
import argparse
import logging
from typing import List, Dict, Any, Optional
import time

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import PDF_DIR, DB_DIR, WEB_CACHE_DIR, TRUSTED_WEBSITES
from src.document_processor.pdf_loader import VedicPDFLoader
from src.document_processor.text_splitter import VedicTextSplitter
from src.knowledge_base.embeddings import get_huggingface_embeddings
from src.knowledge_base.vector_store import VedicVectorStore
from src.qa_system.llm_interface import VedicLLMInterface
from src.qa_system.retriever import VedicRetriever
from src.qa_system.citation import CitationManager
from src.web_scraper.scraper import VedicWebScraper
from src.web_scraper.dynamic_scraper import DynamicVedicScraper
from src.web_scraper.scheduler import ScrapingScheduler
from src.web_scraper.cache_manager import WebCacheManager
from src.utils.exporter import DataExporter
from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger(
    name="vedic_knowledge_ai",
    log_file="vedic_knowledge_ai.log"
)

class VedicKnowledgeAI:
    """Main application class for Vedic Knowledge AI."""
    
    def __init__(self, pdf_dir: str = PDF_DIR, db_dir: str = DB_DIR, cache_dir: str = WEB_CACHE_DIR):
        """Initialize the Vedic Knowledge AI system."""
        self.pdf_dir = pdf_dir
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        
        logger.info("Initializing Vedic Knowledge AI...")
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Vedic Knowledge AI initialized successfully")
    
    def _initialize_components(self):
        """Initialize all system components."""
        # Initialize embedding model
        self.embeddings = get_huggingface_embeddings()
        
        # Initialize vector store
        self.vector_store = VedicVectorStore(
            embedding_function=self.embeddings,
            persist_directory=self.db_dir
        )
        
        # Initialize LLM interface
        self.llm_interface = VedicLLMInterface()
        
        # Initialize retriever
        self.retriever = VedicRetriever(
            vector_store=self.vector_store,
            llm_interface=self.llm_interface
        )
        
        # Initialize citation manager
        self.citation_manager = CitationManager()
        
        # Initialize web cache manager
        self.cache_manager = WebCacheManager(cache_dir=self.cache_dir)
        
        # Initialize web scrapers
        self.web_scraper = VedicWebScraper(cache_dir=self.cache_dir)
        self.dynamic_scraper = DynamicVedicScraper(cache_dir=self.cache_dir)
        
        # Initialize scraping scheduler
        self.scraping_scheduler = ScrapingScheduler(
            vector_store=self.vector_store,
            websites=TRUSTED_WEBSITES
        )
    
    def load_documents(self, limit: Optional[int] = None):
        """Load documents into the vector store."""
        logger.info(f"Loading documents from {self.pdf_dir}...")
        
        # Initialize PDF loader
        pdf_loader = VedicPDFLoader(directory=self.pdf_dir)
        
        # Load PDFs
        documents = pdf_loader.load_all_pdfs(limit=limit)
        
        if not documents:
            logger.warning("No documents loaded")
            return
        
        # Split documents into chunks
        text_splitter = VedicTextSplitter()
        chunks = text_splitter.split_documents(documents)
        
        # Add to vector store
        self.vector_store.add_documents(chunks)
        
        logger.info(f"Loaded {len(documents)} documents ({len(chunks)} chunks)")
    
    def start_scraping(self, immediate: bool = False):
        """Start the scheduled web scraping."""
        logger.info("Starting web scraping scheduler...")
        self.scraping_scheduler.start(immediate=immediate)
    
    def stop_scraping(self):
        """Stop the scheduled web scraping."""
        logger.info("Stopping web scraping scheduler...")
        self.scraping_scheduler.stop()
    
    def add_website(self, url: str, scrape_now: bool = False):
        """Add a website to the scraping list."""
        logger.info(f"Adding website to scraping list: {url}")
        return self.scraping_scheduler.add_website(url, scrape_now=scrape_now)
    
    def scrape_website(self, url: str, bypass_cache: bool = False, is_dynamic: bool = False):
        """Scrape a single website and add to knowledge base."""
        logger.info(f"Scraping website: {url}")
        
        if is_dynamic:
            # Use the dynamic scraper for JavaScript-heavy sites
            documents = self.dynamic_scraper.scrape_url(url, bypass_cache=bypass_cache)
        else:
            # Use the regular scraper
            documents = self.web_scraper.scrape_url(url, bypass_cache=bypass_cache)
        
        if not documents:
            logger.warning(f"No documents extracted from {url}")
            return False
        
        # Add to vector store
        self.vector_store.add_documents(documents)
        
        logger.info(f"Successfully added {len(documents)} chunks from {url} to knowledge base")
        return True
    
    def clear_web_cache(self, all_entries: bool = False):
        """Clear the web cache."""
        if all_entries:
            count = self.cache_manager.clear_all()
            logger.info(f"Cleared all cache entries ({count} files)")
        else:
            count = self.cache_manager.clear_expired()
            logger.info(f"Cleared {count} expired cache entries")
        
        return count
    
    def get_cache_stats(self):
        """Get statistics about the web cache."""
        return self.cache_manager.get_stats()
    
    def answer_question(self, question: str, filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Answer a question using the knowledge base."""
        logger.info(f"Answering question: {question}")
        result = self.retriever.answer_query(question, filter_dict)
        
        # Add formatted citations
        if result.get("documents"):
            result["formatted_citations"] = self.citation_manager.format_citation_list(result["documents"])
        
        return result
    
    def answer_question_with_export(self, question: str, filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Answer a question and export the Q&A pair."""
        # Get the answer
        result = self.answer_question(question, filter_dict)
        
        # Format sources for export
        sources = []
        if "sources" in result:
            sources = [src.get("source", "Unknown") for src in result["sources"]]
        
        # Export the Q&A pair
        export_path = DataExporter.export_qa_log(
            question=question,
            answer=result.get("answer", ""),
            sources=sources
        )
        
        # Add export path to result
        result["export_path"] = export_path
        
        return result
    
    def explain_sanskrit_term(self, term: str) -> Dict[str, Any]:
        """Generate an explanation for a Sanskrit term."""
        logger.info(f"Explaining Sanskrit term: {term}")
        
        # First, try to find relevant documents
        docs = self.retriever.retrieve_documents(
            query=f"Sanskrit term meaning {term}",
            filter_dict=None,
            k=5
        )
        
        context = ""
        if docs:
            context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate explanation using LLM
        explanation = self.llm_interface.explain_sanskrit_term(term, context)
        
        # Format result
        result = {
            "term": term,
            "explanation": explanation,
            "sources": self.citation_manager.format_citations_for_docs(docs) if docs else [],
            "documents": docs
        }
        
        return result
    
    def explain_verse(self, verse: str, reference: Optional[str] = None) -> Dict[str, Any]:
        """Generate an explanation for a verse."""
        query = verse
        if reference:
            query = f"{reference} {verse}"
        
        logger.info(f"Explaining verse: {query}")
        
        # Try to find relevant documents
        docs = self.retriever.retrieve_documents(
            query=query,
            filter_dict=None,
            k=5
        )
        
        context = ""
        if docs:
            context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate explanation using LLM
        explanation = self.llm_interface.explain_verse(verse, reference, context)
        
        # Format result
        result = {
            "verse": verse,
            "reference": reference,
            "explanation": explanation,
            "sources": self.citation_manager.format_citations_for_docs(docs) if docs else [],
            "documents": docs
        }
        
        return result
    
    def export_sanskrit_terms_dictionary(self):
        """Extract and export Sanskrit terms from the knowledge base."""
        logger.info("Exporting Sanskrit terms dictionary")
        
        # Get documents containing Sanskrit
        docs = self.vector_store.filter_by_metadata({"contains_sanskrit": True}, k=1000)
        
        if not docs:
            logger.warning("No Sanskrit content found in the knowledge base")
            return None
        
        # Extract terms
        terms_dict = {}
        for doc in docs:
            sanskrit_terms = doc.metadata.get("sanskrit_terms", [])
            term_definitions = doc.metadata.get("term_definitions", {})
            
            for term in sanskrit_terms:
                if term not in terms_dict:
                    terms_dict[term] = {
                        "definition": term_definitions.get(term, ""),
                        "sources": [self.citation_manager.format_citation(doc)],
                        "examples": []
                    }
                else:
                    # Add source if not already in the list
                    source = self.citation_manager.format_citation(doc)
                    if source not in terms_dict[term]["sources"]:
                        terms_dict[term]["sources"].append(source)
                    
                    # Update definition if empty
                    if not terms_dict[term]["definition"] and term_definitions.get(term):
                        terms_dict[term]["definition"] = term_definitions.get(term)
        
        # Export to file
        if terms_dict:
            export_path = DataExporter.export_sanskrit_terms(terms_dict)
            logger.info(f"Exported {len(terms_dict)} Sanskrit terms to {export_path}")
            return export_path
        else:
            logger.warning("No Sanskrit terms found in the knowledge base")
            return None
    
    def generate_system_report(self):
        """Generate a system report with statistics and information."""
        logger.info("Generating system report")
        
        # Collect vector database statistics
        db_stats = self.vector_store.get_statistics()
        
        # Collect cache statistics
        cache_stats = self.get_cache_stats()
        
        # Collect scraper statistics
        scraper_stats = self.scraping_scheduler.get_status()
        
        # Combined statistics
        stats = {
            "database": db_stats,
            "cache": cache_stats,
            "scraper": scraper_stats,
            "system": {
                "pdf_directory": self.pdf_dir,
                "db_directory": self.db_dir,
                "cache_directory": self.cache_dir,
                "timestamp": time.time()
            }
        }
        
        # Export statistics
        export_path = DataExporter.export_statistics(stats, name="system_report")
        
        return export_path
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the vector database."""
        logger.info("Getting database information")
        
        # Get vector store statistics
        vector_stats = self.vector_store.get_statistics()
        
        return vector_stats


def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(description="Vedic Knowledge AI")
    
    # Define subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Initialize command
    init_parser = subparsers.add_parser("init", help="Initialize the system")
    init_parser.add_argument("--pdf-dir", help="Directory containing PDF files", default=PDF_DIR)
    init_parser.add_argument("--db-dir", help="Directory for vector database", default=DB_DIR)
    init_parser.add_argument("--cache-dir", help="Directory for web cache", default=WEB_CACHE_DIR)
    
    # Load documents command
    load_parser = subparsers.add_parser("load", help="Load documents into the system")
    load_parser.add_argument("--limit", type=int, help="Limit the number of documents to load")
    
    # Scrape website command
    scrape_parser = subparsers.add_parser("scrape", help="Scrape a website")
    scrape_parser.add_argument("url", help="URL to scrape")
    scrape_parser.add_argument("--dynamic", action="store_true", help="Use dynamic scraper for JavaScript-heavy sites")
    scrape_parser.add_argument("--bypass-cache", action="store_true", help="Bypass cache and fetch fresh content")
    
    # Start/stop scraping scheduler
    scrape_scheduler_parser = subparsers.add_parser("scheduler", help="Control the scraping scheduler")
    scrape_scheduler_parser.add_argument("action", choices=["start", "stop", "status"], help="Action to perform")
    scrape_scheduler_parser.add_argument("--immediate", action="store_true", help="Run scraping immediately")
    
    # Cache management command
    cache_parser = subparsers.add_parser("cache", help="Manage web cache")
    cache_parser.add_argument("action", choices=["clear", "stats"], help="Action to perform")
    cache_parser.add_argument("--all", action="store_true", help="Clear all cache entries (not just expired)")
    
    # Answer question command
    answer_parser = subparsers.add_parser("answer", help="Answer a question")
    answer_parser.add_argument("question", help="Question to answer")
    answer_parser.add_argument("--export", action="store_true", help="Export the Q&A pair")
    
    # Explain Sanskrit term command
    explain_term_parser = subparsers.add_parser("explain-term", help="Explain a Sanskrit term")
    explain_term_parser.add_argument("term", help="Sanskrit term to explain")
    
    # Explain verse command
    explain_verse_parser = subparsers.add_parser("explain-verse", help="Explain a verse")
    explain_verse_parser.add_argument("verse", help="Verse to explain")
    explain_verse_parser.add_argument("--reference", help="Reference (e.g., 'Bhagavad Gita 2.47')")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export data")
    export_parser.add_argument("type", choices=["terms", "report"], help="Type of data to export")
    
    # Get database info command
    info_parser = subparsers.add_parser("info", help="Get database information")
    
    # Interactive mode
    interactive_parser = subparsers.add_parser("interactive", help="Start interactive mode")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize the system
    if args.command == "init":
        # Create directories if they don't exist
        os.makedirs(args.pdf_dir, exist_ok=True)
        os.makedirs(args.db_dir, exist_ok=True)
        os.makedirs(args.cache_dir, exist_ok=True)
        
        print(f"Initialized system with PDF directory: {args.pdf_dir}")
        print(f"Vector database directory: {args.db_dir}")
        print(f"Web cache directory: {args.cache_dir}")
    
    # Create the AI system
    ai = None
    if args.command in ["load", "scrape", "scheduler", "cache", "answer", "explain-term", "explain-verse", "export", "info", "interactive"]:
        ai = VedicKnowledgeAI()
    
    # Execute the command
    if args.command == "load":
        ai.load_documents(limit=args.limit)
        print(f"Loaded documents from {ai.pdf_dir}")
    
    elif args.command == "scrape":
        success = ai.scrape_website(
            args.url, 
            bypass_cache=args.bypass_cache,
            is_dynamic=args.dynamic
        )
        if success:
            print(f"Successfully scraped {args.url}")
        else:
            print(f"Failed to scrape {args.url}")
    
    elif args.command == "scheduler":
        if args.action == "start":
            ai.start_scraping(immediate=args.immediate)
            print("Started scraping scheduler")
        elif args.action == "stop":
            ai.stop_scraping()
            print("Stopped scraping scheduler")
        elif args.action == "status":
            status = ai.scraping_scheduler.get_status()
            print("Scraping scheduler status:")
            print(f"Running: {status['is_running']}")
            print(f"Websites: {', '.join(status['websites'])}")
            print(f"Interval: {status['scraping_interval']} seconds")
    
    elif args.command == "cache":
        if args.action == "clear":
            count = ai.clear_web_cache(all_entries=args.all)
            if args.all:
                print(f"Cleared all cache entries ({count} files)")
            else:
                print(f"Cleared {count} expired cache entries")
        elif args.action == "stats":
            stats = ai.get_cache_stats()
            print("\nWeb Cache Statistics:")
            print(f"Total entries: {stats['total_entries']}")
            print(f"Total size: {stats['total_size_mb']} MB")
            print(f"Hit rate: {stats['hit_rate']}%")
            print(f"Domains: {len(stats['domains'])}")
            print("\nTop domains:")
            
            # Sort domains by entry count
            sorted_domains = sorted(stats['domains'].items(), key=lambda x: x[1], reverse=True)
            for i, (domain, count) in enumerate(sorted_domains[:5]):
                print(f"  {i+1}. {domain}: {count} entries")
    
    elif args.command == "answer":
        if args.export:
            result = ai.answer_question_with_export(args.question)
            print("\n" + "="*50)
            print(f"Question: {args.question}")
            print("="*50)
            print(f"Answer: {result['answer']}")
            print("-"*50)
            if "formatted_citations" in result:
                print(result["formatted_citations"])
            print("="*50)
            print(f"Exported to: {result['export_path']}")
            print("="*50 + "\n")
        else:
            result = ai.answer_question(args.question)
            print("\n" + "="*50)
            print(f"Question: {args.question}")
            print("="*50)
            print(f"Answer: {result['answer']}")
            print("-"*50)
            if "formatted_citations" in result:
                print(result["formatted_citations"])
            print("="*50 + "\n")
    
    elif args.command == "explain-term":
        result = ai.explain_sanskrit_term(args.term)
        print("\n" + "="*50)
        print(f"Sanskrit Term: {args.term}")
        print("="*50)
        print(result["explanation"])
        print("-"*50)
        if result["sources"]:
            print("Sources:")
            for i, source in enumerate(result["sources"]):
                print(f"{i+1}. {source}")
        print("="*50 + "\n")
    
    elif args.command == "explain-verse":
        result = ai.explain_verse(args.verse, args.reference)
        print("\n" + "="*50)
        print(f"Verse: {args.verse}")
        if args.reference:
            print(f"Reference: {args.reference}")
        print("="*50)
        print(result["explanation"])
        print("-"*50)
        if result["sources"]:
            print("Sources:")
            for i, source in enumerate(result["sources"]):
                print(f"{i+1}. {source}")
        print("="*50 + "\n")
    
    elif args.command == "export":
        if args.type == "terms":
            export_path = ai.export_sanskrit_terms_dictionary()
            if export_path:
                print(f"Exported Sanskrit terms dictionary to {export_path}")
            else:
                print("No Sanskrit terms found in the knowledge base")
        elif args.type == "report":
            export_path = ai.generate_system_report()
            print(f"Generated system report at {export_path}")
    
    elif args.command == "info":
        info = ai.get_database_info()
        print("\n" + "="*50)
        print("Database Information:")
        print(f"Document count: {info['document_count']}")
        print(f"Collection name: {info['collection_name']}")
        print(f"Directory: {info['persist_directory']}")
        print("="*50 + "\n")
    
    elif args.command == "interactive":
        print("\nWelcome to Vedic Knowledge AI Interactive Mode")
        print("Type 'help' for a list of commands, or 'exit' to quit\n")
        
        while True:
            try:
                user_input = input("> ").strip()
                
                if user_input.lower() in ["exit", "quit", "q"]:
                    break
                
                if user_input.lower() in ["help", "h", "?"]:
                    print("\nAvailable commands:")
                    print("  ask <question>         - Ask a question")
                    print("  term <sanskrit term>   - Explain a Sanskrit term")
                    print("  verse <verse text>     - Explain a verse")
                    print("  scrape <url>           - Scrape a website")
                    print("  dynamic <url>          - Scrape a JS-heavy website")
                    print("  cache stats            - Show cache statistics")
                    print("  cache clear            - Clear expired cache entries")
                    print("  export terms           - Export Sanskrit terms dictionary")
                    print("  export report          - Generate system report")
                    print("  info                   - Show database information")
                    print("  exit                   - Exit interactive mode")
                    print()
                    continue
                
                if user_input.lower().startswith("ask "):
                    question = user_input[4:].strip()
                    if question:
                        result = ai.answer_question_with_export(question)
                        print("\nAnswer:")
                        print(result["answer"])
                        if "formatted_citations" in result:
                            print("\nSources:")
                            print(result["formatted_citations"])
                        print(f"\nExported to: {result['export_path']}")
                    else:
                        print("Please provide a question")
                
                elif user_input.lower().startswith("term "):
                    term = user_input[5:].strip()
                    if term:
                        result = ai.explain_sanskrit_term(term)
                        print("\nExplanation:")
                        print(result["explanation"])
                        if result["sources"]:
                            print("\nSources:")
                            for i, source in enumerate(result["sources"]):
                                print(f"{i+1}. {source}")
                    else:
                        print("Please provide a Sanskrit term")
                
                elif user_input.lower().startswith("verse "):
                    verse = user_input[6:].strip()
                    if verse:
                        result = ai.explain_verse(verse)
                        print("\nExplanation:")
                        print(result["explanation"])
                        if result["sources"]:
                            print("\nSources:")
                            for i, source in enumerate(result["sources"]):
                                print(f"{i+1}. {source}")
                    else:
                        print("Please provide a verse")
                
                elif user_input.lower().startswith("scrape "):
                    url = user_input[7:].strip()
                    if url:
                        print(f"Scraping {url}...")
                        success = ai.scrape_website(url)
                        if success:
                            print(f"Successfully scraped {url}")
                        else:
                            print(f"Failed to scrape {url}")
                    else:
                        print("Please provide a URL")
                
                elif user_input.lower().startswith("dynamic "):
                    url = user_input[8:].strip()
                    if url:
                        print(f"Scraping dynamic site {url}...")
                        success = ai.scrape_website(url, is_dynamic=True)
                        if success:
                            print(f"Successfully scraped {url}")
                        else:
                            print(f"Failed to scrape {url}")
                    else:
                        print("Please provide a URL")
                
                elif user_input.lower() == "cache stats":
                    stats = ai.get_cache_stats()
                    print("\nWeb Cache Statistics:")
                    print(f"Total entries: {stats['total_entries']}")
                    print(f"Total size: {stats['total_size_mb']} MB")
                    print(f"Hit rate: {stats['hit_rate']}%")
                    
                    # Print domain stats
                    print("\nTop domains:")
                    sorted_domains = sorted(stats['domains'].items(), key=lambda x: x[1], reverse=True)
                    for i, (domain, count) in enumerate(sorted_domains[:5]):
                        print(f"  {i+1}. {domain}: {count} entries")
                
                elif user_input.lower() == "cache clear":
                    count = ai.clear_web_cache()
                    print(f"Cleared {count} expired cache entries")
                
                elif user_input.lower() == "export terms":
                    export_path = ai.export_sanskrit_terms_dictionary()
                    if export_path:
                        print(f"Exported Sanskrit terms dictionary to {export_path}")
                    else:
                        print("No Sanskrit terms found in the knowledge base")
                
                elif user_input.lower() == "export report":
                    export_path = ai.generate_system_report()
                    print(f"Generated system report at {export_path}")
                
                elif user_input.lower() == "info":
                    info = ai.get_database_info()
                    print("\nDatabase Information:")
                    print(f"Document count: {info['document_count']}")
                    print(f"Collection name: {info['collection_name']}")
                    print(f"Directory: {info['persist_directory']}")
                
                else:
                    print("Unknown command. Type 'help' for a list of commands")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {str(e)}")
        
        print("\nExiting interactive mode")


def lookup_sanskrit_term(self, term: str, bypass_cache: bool = False) -> Dict[str, Any]:
    """Look up a Sanskrit term on Vedabase and return the data."""
    from src.web_scraper.scraper import scrape_vedabase_term
    
    term_data = scrape_vedabase_term(term, bypass_cache=bypass_cache)
    
    if term_data and term_data.get("occurrences"):
        # Add to vector store if we have occurrences
        self._add_term_to_knowledge_base(term_data)
    
    return term_data

def _add_term_to_knowledge_base(self, term_data: Dict[str, Any]) -> None:
    """Add term data to the knowledge base."""
    from langchain.docstore.document import Document
    
    # Create a document for the term definition
    term = term_data.get("term", "")
    devanagari = term_data.get("devanagari", "")
    definition = term_data.get("definition", "")
    
    # Prepare content and metadata
    content = f"Term: {term}\n"
    if devanagari:
        content += f"Devanagari: {devanagari}\n"
    content += f"Definition: {definition}\n\n"
    
    # Add occurrences
    for occ in term_data.get("occurrences", []):
        content += f"Occurrence in {occ['reference']}: {occ['text']}\n"
    
    # Create metadata
    metadata = {
        "source": "Vedabase.io",
        "type": "sanskrit_term",
        "term": term,
        "devanagari": devanagari,
        "contains_sanskrit": True,
        "sanskrit_terms": [term]
    }
    
    # Create document
    doc = Document(page_content=content, metadata=metadata)
    
    # Add to vector store
    self.vector_store.add_documents([doc])
    
    logger.info(f"Added term {term} to knowledge base")

if __name__ == "__main__":
    main()