# src/document_processor/sanskrit_processor.py
# Applied changes: Item 8 (Simplify WordNet Init), Item 11 (Type Hinting)
"""
Sanskrit text processing utilities for Vedic Knowledge AI.
Handles Sanskrit-specific text normalization, tokenization and processing.
Uses optional dependencies: indic-nlp-library and pyiwn (IndoWordNet).
"""
import re
import logging
from typing import List, Dict, Any, Optional

# Import indic-nlp-library components safely
try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    from indicnlp.tokenize import indic_tokenize
    from indicnlp.transliterate import unicode_transliterate
    # Define language code for Sanskrit within indic_nlp context if needed
    INDIC_NLP_LANG = 'sa' # Check indic-nlp docs for correct Sanskrit code
    HAS_INDIC_NLP = True
except ImportError:
    HAS_INDIC_NLP = False
    logging.warning("indic-nlp-library not found. Sanskrit processing capabilities (normalization, transliteration) will be limited. Install with: pip install indic-nlp-library")

# Try to import pyiwn for Sanskrit WordNet safely
try:
    import pyiwn
    # Check if Language enum exists and has SANSKRIT attribute
    if hasattr(pyiwn, 'Language') and hasattr(pyiwn.Language, 'SANSKRIT'):
        PYIWN_LANG = pyiwn.Language.SANSKRIT
    else:
        # Fallback to string code if enum is not available or doesn't have SANSKRIT
        # Check pyiwn documentation for the recommended language code ('san', 'skt', etc.)
        PYIWN_LANG = 'san' 
    HAS_PYIWN = True
except ImportError:
    HAS_PYIWN = False
    logging.warning("pyiwn (IndoWordNet) not found. Sanskrit WordNet capabilities will be limited. Install with: pip install pyiwn")

# Configure logging
logger = logging.getLogger(__name__)

class SanskritProcessor:
    """
    Provides utilities for processing text containing Sanskrit, including
    detection, normalization, transliteration, term extraction, and definition lookup (via IndoWordNet).
    """
    def __init__(self):
        """Initialize the Sanskrit processor and its components."""
        self.has_indic_nlp = HAS_INDIC_NLP
        self.has_pyiwn = HAS_PYIWN
        self.sanskrit_wordnet: Optional[pyiwn.IndoWordNet] = None
        self.indic_normalizer: Optional[IndicNormalizerFactory.Normalizer] = None

        # Initialize Indic NLP Normalizer if available
        if self.has_indic_nlp:
            try:
                # Create normalizer factory for Sanskrit
                factory = IndicNormalizerFactory()
                # Check available normalizers for the language
                self.indic_normalizer = factory.get_normalizer(INDIC_NLP_LANG)
                logger.info(f"Indic NLP normalizer for '{INDIC_NLP_LANG}' initialized.")
            except Exception as e:
                logger.warning(f"Could not initialize Indic NLP normalizer: {e}. Normalization will be basic.")
                self.has_indic_nlp = False # Disable if initialization fails

        # Initialize Sanskrit WordNet if available
        if self.has_pyiwn:
            try:
                # Attempt simplified initialization using the determined language code/enum
                self.sanskrit_wordnet = pyiwn.IndoWordNet(lang=PYIWN_LANG)
                logger.info(f"Sanskrit WordNet initialized successfully using lang='{PYIWN_LANG}'.")
            except Exception as e:
                # Log specific error if possible (e.g., data files not found)
                logger.warning(f"Could not initialize Sanskrit WordNet with lang='{PYIWN_LANG}': {e}. WordNet features disabled.")
                self.has_pyiwn = False # Disable if initialization fails
        
        # Compile regex patterns for Sanskrit detection (moved to instance attributes)
        # Pattern to detect Devanagari script characters (common range)
        self.devanagari_pattern = re.compile(r'[ऀ-ॿ]') # More general Devanagari range
        # Pattern for common IAST transliteration characters with diacritics
        self.iast_pattern = re.compile(r'[āīūṛṝḷḹṁḥśṣṭḍṇṅñ]') # Added more IAST chars
        # Common Sanskrit/Indic philosophical terms pattern (case-insensitive)
        self.common_terms_pattern = re.compile(
            r'\b(dharma|karma|yoga|brahman|atman|moksha|samsara|vedanta|bhakti|jnana|bhagavad|gita|upanishad|veda|purana|sutra|acharya|guru|mantra|puja|yajna|sampradaya)\b', 
            re.IGNORECASE
        )
        
        logger.info(f"Sanskrit processor initialized (IndicNLP: {self.has_indic_nlp}, PyIWN: {self.has_pyiwn})")
        
    
    def contains_sanskrit(self, text: str) -> bool:
        """
        Check if the text likely contains Sanskrit content based on script or common terms.

        Args:
            text (str): The input text.

        Returns:
            bool: True if Sanskrit content is detected, False otherwise.
        """
        if not text:
            return False
            
        # Check for Devanagari script
        if self.devanagari_pattern.search(text):
            logger.debug("Devanagari script detected.")
            return True
        
        # Check for IAST transliteration
        if self.iast_pattern.search(text):
            logger.debug("IAST characters detected.")
            return True
        
        # Check for common Sanskrit terms
        if self.common_terms_pattern.search(text):
            logger.debug("Common Sanskrit terms detected.")
            return True
            
        return False
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize Sanskrit text using indic-nlp-library if available, otherwise basic normalization.

        Args:
            text (str): The input text.

        Returns:
            str: The normalized text.
        """
        if not text:
            return ""
            
        # Use indic-nlp normalizer if available and text likely contains Devanagari
        if self.has_indic_nlp and self.indic_normalizer and self.devanagari_pattern.search(text):
            try:
                normalized_text = self.indic_normalizer.normalize(text)
                logger.debug("Normalized text using Indic NLP.")
                return normalized_text
            except Exception as e:
                logger.error(f"Error normalizing Sanskrit text using Indic NLP: {e}")
                # Fallback to basic normalization on error
                return self._basic_normalize(text)
        else:
            # Apply basic normalization if Indic NLP is not used or not applicable
             return self._basic_normalize(text)

    def _basic_normalize(self, text: str) -> str:
        """Basic normalization for Sanskrit text (primarily for IAST)."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Basic replacements for common IAST variations (can be expanded)
        # This is a very simple approach and might not be linguistically correct in all cases.
        replacements = {
            'ā': 'a', 'ī': 'i', 'ū': 'u', 'ṛ': 'ri', 'ṝ': 'ri', 
            'ḷ': 'li', 'ḹ': 'li', #'ē': 'e', 'ō': 'o', # Keep e, o as they are distinct vowels
            'ṁ': 'm', 'ḥ': 'h', 'ṅ': 'n', 'ñ': 'n', 'ṭ': 't',
            'ḍ': 'd', 'ṇ': 'n', 'ś': 'sh', 'ṣ': 'sh'
        }
        
        normalized_text = text
        for orig, repl in replacements.items():
            normalized_text = normalized_text.replace(orig, repl)
        
        logger.debug("Applied basic text normalization.")
        return normalized_text
    
    def transliterate(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Transliterate text between scripts using indic-nlp-library.

        Args:
            text (str): The text to transliterate.
            source_lang (str): Source language/script code (e.g., 'hi' for Devanagari, 'en' for IAST/Roman).
            target_lang (str): Target language/script code.

        Returns:
            Optional[str]: Transliterated text, or None if transliteration fails or indic-nlp is unavailable.
        """
        if not self.has_indic_nlp:
            logger.warning("indic-nlp-library is required for transliteration.")
            return None # Return None instead of original text to indicate failure
            
        try:
            # Use the transliteration module
            transliterated_text = unicode_transliterate.UnicodeIndicTransliterator.transliterate(
                text, source_lang, target_lang
            )
            logger.debug(f"Transliterated text from {source_lang} to {target_lang}.")
            return transliterated_text
        except Exception as e:
            logger.error(f"Error during transliteration from {source_lang} to {target_lang}: {e}")
            return None # Return None on error

    def transliterate_to_devanagari(self, text: str) -> Optional[str]:
        """Transliterate IAST/Roman text to Devanagari."""
        # Assuming source is Roman script, typically represented by 'en' in indic-nlp
        # Target is Devanagari, typically represented by 'hi' or 'sa'
        return self.transliterate(text, "en", INDIC_NLP_LANG) 

    def transliterate_to_iast(self, text: str) -> Optional[str]:
        """Transliterate Devanagari text to IAST/Roman."""
         # Source is Devanagari ('hi' or 'sa'), target is Roman ('en')
        return self.transliterate(text, INDIC_NLP_LANG, "en")
    
    def extract_sanskrit_terms(self, text: str) -> List[str]:
        """
        Extract potential Sanskrit terms from the text based on script and common word list.

        Args:
            text (str): The input text.

        Returns:
            List[str]: A list of unique potential Sanskrit terms found.
        """
        terms: set[str] = set() # Use a set for efficient uniqueness
        
        # Extract words containing Devanagari characters
        # Regex needs refinement to better capture word boundaries with punctuation
        devanagari_words = re.findall(r'\b[ऀ-ॿ]+[ऀ-ॿ\w]*\b', text) # Find words starting with Devanagari
        terms.update(word.strip('.,!?;:"\'()[]{}') for word in devanagari_words) # Clean punctuation
        
        # Extract words containing specific IAST characters
        iast_words = re.findall(r'\b\w*[āīūṛṝḷḹṁḥśṣṭḍṇṅñ]+\w*\b', text) # Find words with IAST chars
        terms.update(word.strip('.,!?;:"\'()[]{}') for word in iast_words)

        # Extract common predefined terms
        common_terms_found = self.common_terms_pattern.findall(text)
        terms.update(term.lower() for term in common_terms_found) # Store common terms lowercase
        
        # Convert set back to list
        extracted_terms = list(terms)
        logger.debug(f"Extracted {len(extracted_terms)} potential Sanskrit terms.")
        return extracted_terms
    
    def get_term_definition(self, term: str) -> Optional[str]:
        """
        Get a definition for a Sanskrit term using IndoWordNet if available, 
        otherwise fallback to a basic hardcoded dictionary.

        Args:
            term (str): The Sanskrit term (preferably normalized or in a standard form).

        Returns:
            Optional[str]: The definition string, or None if not found.
        """
        # Try IndoWordNet first if available
        if self.has_pyiwn and self.sanskrit_wordnet:
            try:
                # Lemmatization might be needed for better lookup (requires pyiwn integration)
                # For now, directly look up the term
                synsets = self.sanskrit_wordnet.synsets(term)
                if synsets:
                    # Return the definition of the first synset found
                    definition = synsets[0].definition()
                    logger.debug(f"Found WordNet definition for '{term}'.")
                    return definition
            except Exception as e:
                # Log lookup errors, could be due to term format or WordNet issues
                logger.debug(f"Error getting definition for '{term}' from WordNet: {e}")
        
        # Fallback: Look for common known terms (case-insensitive)
        common_definitions: Dict[str, str] = {
            "dharma": "Righteous duty, virtue, intrinsic nature, or the cosmic law.",
            "karma": "Action, work, or deed, and its consequential principle of cause and effect.",
            "yoga": "Union, communion, or discipline; practices for spiritual realization.",
            "atman": "The true Self, individual soul, or principle of life.",
            "brahman": "The ultimate reality, the Absolute, the supreme consciousness.",
            "moksha": "Liberation, release, or emancipation from the cycle of birth and death (samsara).",
            "samsara": "The cycle of birth, death, and rebirth; the phenomenal world.",
            "bhakti": "Devotion, worship, or loving service to the Divine.",
            "jnana": "Knowledge, wisdom, especially spiritual knowledge.",
            "acharya": "Spiritual teacher or guide who teaches by example.",
            "mantra": "Sacred utterance, sound, or syllable used in meditation or ritual.",
            # Add more common terms as needed
        }
        
        term_lower = term.lower()
        definition = common_definitions.get(term_lower)
        if definition:
             logger.debug(f"Found fallback definition for '{term}'.")
        else:
             logger.debug(f"No definition found for '{term}'.")
             
        return definition
    
    def process_document_metadata(self, text: str) -> Dict[str, Any]:
        """
        Analyze text to extract metadata relevant to Sanskrit content.

        Args:
            text (str): The input text document.

        Returns:
            Dict[str, Any]: A dictionary containing metadata like 'contains_sanskrit' 
                            and 'sanskrit_terms'. Definitions are not looked up here
                            to keep it lightweight for bulk processing.
        """
        contains_sanskrit = self.contains_sanskrit(text)
        sanskrit_terms = []
        if contains_sanskrit:
            sanskrit_terms = self.extract_sanskrit_terms(text)
            
        result: Dict[str, Any] = {
            # "normalized_text": self.normalize_text(text), # Normalization can be heavy, apply only if needed
            "contains_sanskrit": contains_sanskrit,
            "sanskrit_terms": sanskrit_terms,
            # "term_definitions": {} # Definitions lookup is separate
        }
                
        return result