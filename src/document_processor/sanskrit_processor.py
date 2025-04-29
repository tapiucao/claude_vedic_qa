"""
Sanskrit text processing utilities for Vedic Knowledge AI.
Handles Sanskrit-specific text normalization, tokenization and processing.
"""
import re
import logging
from typing import List, Dict, Any, Optional

# Import indic-nlp-library components
try:
    from indicnlp.normalize import indic_normalize
    from indicnlp.tokenize import indic_tokenize
    from indicnlp.transliterate import unicode_transliterate
    HAS_INDIC_NLP = True
except ImportError:
    HAS_INDIC_NLP = False
    logging.warning("indic-nlp-library not found. Sanskrit processing capabilities will be limited.")

# Try to import pyiwn for Sanskrit WordNet
try:
    import pyiwn
    HAS_PYIWN = True
except ImportError:
    HAS_PYIWN = False
    logging.warning("pyiwn not found. Sanskrit WordNet capabilities will be limited.")

# Configure logging
logger = logging.getLogger(__name__)

class SanskritProcessor:
    def __init__(self):
        """Initialize the Sanskrit processor."""
        self.has_indic_nlp = HAS_INDIC_NLP
        self.sanskrit_wordnet = None
        
        # Try to initialize Sanskrit WordNet but make it optional
        if HAS_PYIWN:
            try:
                # Different initialization approach
                self.sanskrit_wordnet = pyiwn.IndoWordNet(lang='san')
                logger.info("Sanskrit WordNet initialized successfully")
                self.has_pyiwn = True
            except Exception as e:
                logger.warning(f"Could not initialize Sanskrit WordNet: {e}. Will proceed without it.")
                self.has_pyiwn = False
        
        # Compile regex patterns for Sanskrit detection
        self._compile_patterns()
        
        logger.info("Sanskrit processor initialized")
    
    def _compile_patterns(self):
        """Compile regex patterns for Sanskrit processing."""
        # Pattern to detect Devanagari script
        self.devanagari_pattern = re.compile(r'[ऄ-औक-ह\u093Cा-ौ्ंःँॐॠऱऴ]')
        
        # Pattern for IAST transliteration characters
        self.iast_pattern = re.compile(r'[āīūṛṝḷḹēōṃḥṅñṭḍṇśṣ]')
        
        # Common Sanskrit terms pattern (basic)
        self.common_terms_pattern = re.compile(
            r'\b(dharma|karma|yoga|brahman|atman|moksha|samsara|vedanta|bhakti|jnana|bhagavad|gita|upanishad|veda|purana|sutra)\b', 
            re.IGNORECASE
        )
    
    def contains_sanskrit(self, text: str) -> bool:
        """Check if the text contains Sanskrit content."""
        # Check for Devanagari script
        if self.devanagari_pattern.search(text):
            return True
        
        # Check for IAST transliteration
        if self.iast_pattern.search(text):
            return True
        
        # Check for common Sanskrit terms
        if self.common_terms_pattern.search(text):
            return True
        
        return False
    
    def normalize_text(self, text: str) -> str:
        """Normalize Sanskrit text."""
        if not text:
            return ""
        
        # Check if text contains Sanskrit
        if not self.contains_sanskrit(text):
            return text
        
        # Use indic-nlp normalizer if available
        if self.has_indic_nlp and self.normalizer:
            try:
                # Check if text contains Devanagari script
                if self.devanagari_pattern.search(text):
                    normalized_text = self.normalizer.normalize(text)
                    return normalized_text
            except Exception as e:
                logger.error(f"Error normalizing Sanskrit text: {e}")
        
        # Fallback to basic normalization
        return self._basic_normalize(text)
    
    def _basic_normalize(self, text: str) -> str:
        """Basic normalization for Sanskrit text."""
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize common variations of diacritical marks
        replacements = {
            'ā': 'a', 'ī': 'i', 'ū': 'u', 'ṛ': 'ri', 'ṝ': 'ri', 
            'ḷ': 'li', 'ḹ': 'li', 'ē': 'e', 'ō': 'o',
            'ṃ': 'm', 'ḥ': 'h', 'ṅ': 'n', 'ñ': 'n', 'ṭ': 't',
            'ḍ': 'd', 'ṇ': 'n', 'ś': 'sh', 'ṣ': 'sh'
        }
        
        for orig, repl in replacements.items():
            text = text.replace(orig, repl)
        
        return text
    
    def transliterate_to_devanagari(self, text: str) -> str:
        """Transliterate IAST text to Devanagari."""
        if not self.has_indic_nlp:
            logger.warning("indic-nlp-library is required for transliteration")
            return text
        
        try:
            if self.iast_pattern.search(text):
                # Convert IAST to Devanagari
                devanagari_text = unicode_transliterate.UnicodeIndicTransliterator.transliterate(
                    text, "en", "hi"
                )
                return devanagari_text
            return text
        except Exception as e:
            logger.error(f"Error in transliteration: {e}")
            return text
    
    def transliterate_to_iast(self, text: str) -> str:
        """Transliterate Devanagari text to IAST."""
        if not self.has_indic_nlp:
            logger.warning("indic-nlp-library is required for transliteration")
            return text
        
        try:
            if self.devanagari_pattern.search(text):
                # Convert Devanagari to IAST
                iast_text = unicode_transliterate.UnicodeIndicTransliterator.transliterate(
                    text, "hi", "en"
                )
                return iast_text
            return text
        except Exception as e:
            logger.error(f"Error in transliteration: {e}")
            return text
    
    def extract_sanskrit_terms(self, text: str) -> List[str]:
        """Extract Sanskrit terms from the text."""
        terms = []
        
        # Extract terms in Devanagari
        if self.devanagari_pattern.search(text):
            # Find all words containing Devanagari characters
            devanagari_words = re.findall(r'\b[\w]*[ऄ-ह़ािीुूृॄेैोौ्ॐंःँॠऱऴक़-य़][\w]*\b', text)
            terms.extend(devanagari_words)
        
        # Extract IAST transliterated terms
        if self.iast_pattern.search(text):
            # Find all words containing IAST characters
            iast_words = re.findall(r'\b[\w]*[āīūṛṝḷḹēōṃḥṅñṭḍṇśṣ][\w]*\b', text)
            terms.extend(iast_words)
        
        # Extract common Sanskrit terms
        common_terms = self.common_terms_pattern.findall(text)
        terms.extend(common_terms)
        
        # Remove duplicates and return
        return list(set(terms))
    
    def get_term_definition(self, term: str) -> Optional[str]:
        """Get definition for a Sanskrit term."""
        # Try WordNet first if available
        if self.has_pyiwn and self.sanskrit_wordnet:
            try:
                synsets = self.sanskrit_wordnet.synsets(term)
                if synsets:
                    return synsets[0].definition()
            except Exception as e:
                logger.debug(f"Error getting definition from WordNet: {e}")
        
        # Fallback: Look for common known terms
        common_definitions = {
            "dharma": "Righteous duty, virtue, or the natural law",
            "karma": "Action or deed and its consequences",
            "yoga": "Union or yoking with the divine, spiritual discipline",
            "atman": "Soul or the true self",
            "brahman": "The ultimate reality or universal consciousness",
            # Add more common terms
        }
        
        # Case-insensitive lookup
        term_lower = term.lower()
        return common_definitions.get(term_lower)
    
    def process_document(self, text: str) -> Dict[str, Any]:
        """Process a document with Sanskrit content."""
        result = {
            "normalized_text": self.normalize_text(text),
            "contains_sanskrit": self.contains_sanskrit(text),
            "sanskrit_terms": self.extract_sanskrit_terms(text),
            "term_definitions": {}
        }
        
        # Add definitions for Sanskrit terms if WordNet is available
        if self.has_pyiwn and self.iwn:
            for term in result["sanskrit_terms"]:
                definition = self.get_term_definition(term)
                if definition:
                    result["term_definitions"][term] = definition
        
        return result