"""
API enhancements for chapter-based queries in Vedic Knowledge AI.
These functions should be added to api.py
"""

# New API models
class ChapterRequest(BaseModel):
    book: Optional[str] = Field(None, description="Optional book name to filter by")
    chapter: int = Field(..., description="Chapter number to filter by")
    limit: Optional[int] = Field(100, description="Maximum number of results to return")

class ChapterSummaryRequest(BaseModel):
    book: Optional[str] = Field(None, description="Optional book name to filter by")
    chapter: int = Field(..., description="Chapter number to summarize")

class ChapterBasedQuestionRequest(BaseModel):
    question: str = Field(..., description="The question to answer")
    book: Optional[str] = Field(None, description="Optional book name to filter by")
    chapter: int = Field(..., description="Chapter to search in")

# New API endpoints
@app.get("/chapters")
async def list_chapters():
    """List all available chapters."""
    try:
        chapters = ai.get_available_chapters()
        return {"chapters": chapters}
    except Exception as e:
        logger.error(f"Error listing chapters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chapters/documents")
async def get_documents_by_chapter(request: ChapterRequest):
    """Get documents from a specific chapter."""
    try:
        documents = ai.get_documents_by_chapter(
            chapter=request.chapter,
            book=request.book,
            limit=request.limit
        )
        
        # Clean up the response (remove document objects)
        result = {
            "chapter": request.chapter,
            "book": request.book,
            "document_count": len(documents),
            "documents": [
                {
                    "page": doc.metadata.get("page", ""),
                    "chapter_reference": doc.metadata.get("chapter_reference", ""),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                for doc in documents[:10]  # Limit preview to 10 documents
            ]
        }
        
        return result
    except Exception as e:
        logger.error(f"Error getting documents by chapter: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chapters/summary")
async def get_chapter_summary(request: ChapterSummaryRequest):
    """Get a summary of a specific chapter."""
    try:
        result = ai.get_chapter_summary(
            chapter=request.chapter,
            book=request.book
        )
        
        # Clean up the response (remove document objects)
        if "documents" in result:
            del result["documents"]
        
        return result
    except Exception as e:
        logger.error(f"Error getting chapter summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chapters/answer")
async def answer_chapter_based_question(request: ChapterBasedQuestionRequest):
    """Answer a question based on a specific chapter."""
    try:
        # Create filter dict for the chapter
        filter_dict = {
            "chapter": request.chapter
        }
        
        if request.book:
            filter_dict["title"] = request.book
        
        # Use the existing answer_question method with the chapter filter
        result = ai.answer_question(request.question, filter_dict)
        
        # Clean up the response (remove document objects)
        if "documents" in result:
            del result["documents"]
        
        return result
    except Exception as e:
        logger.error(f"Error answering chapter-based question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add these methods to the VedicKnowledgeAI class
def get_available_chapters(self) -> List[Dict[str, Any]]:
    """Get a list of all available chapters in the knowledge base."""
    try:
        # Query the vector store for documents with chapter metadata
        all_chapters = {}
        
        # Use vector store's collection to query for unique chapter numbers
        chapters_query = """
        SELECT DISTINCT 
            collection_id, document_id, metadata_chapter as chapter, 
            metadata_title as book, metadata_chapter_reference as reference
        FROM embedding 
        WHERE metadata_chapter IS NOT NULL
        """
        
        results = self.vector_store.vector_store._collection.client._system_client._postgres.execute(chapters_query)
        
        # Process the results
        for row in results:
            chapter = row.get('chapter')
            book = row.get('book', 'Unknown')
            reference = row.get('reference')
            
            key = f"{book}_{chapter}"
            if key not in all_chapters:
                all_chapters[key] = {
                    "book": book,
                    "chapter": chapter,
                    "reference": reference
                }
        
        # Convert to list
        chapter_list = list(all_chapters.values())
        
        # Sort by book and chapter
        chapter_list.sort(key=lambda x: (x.get('book', ''), x.get('chapter', 0)))
        
        return chapter_list
    except Exception as e:
        logger.error(f"Error getting available chapters: {str(e)}")
        # Fallback method if direct DB query fails
        return self._fallback_get_chapters()

def _fallback_get_chapters(self) -> List[Dict[str, Any]]:
    """Fallback method to get chapters by scanning documents."""
    try:
        # Sample random documents to find chapters
        sample_docs = self.vector_store.similarity_search(
            query="chapter",
            k=1000
        )
        
        # Extract chapter info
        all_chapters = {}
        for doc in sample_docs:
            if "chapter" in doc.metadata:
                chapter = doc.metadata.get("chapter")
                book = doc.metadata.get("title", "Unknown")
                reference = doc.metadata.get("chapter_reference")
                
                key = f"{book}_{chapter}"
                if key not in all_chapters:
                    all_chapters[key] = {
                        "book": book,
                        "chapter": chapter,
                        "reference": reference
                    }
        
        # Convert to list and sort
        chapter_list = list(all_chapters.values())
        chapter_list.sort(key=lambda x: (x.get('book', ''), x.get('chapter', 0)))
        
        return chapter_list
    except Exception as e:
        logger.error(f"Error in fallback get chapters: {str(e)}")
        return []

def get_documents_by_chapter(self, chapter: int, book: Optional[str] = None, limit: int = 100) -> List[Document]:
    """Get documents from a specific chapter."""
    return self.retriever.get_documents_by_chapter(chapter, book, limit)

def get_chapter_summary(self, chapter: int, book: Optional[str] = None) -> Dict[str, Any]:
    """Generate a summary for a specific chapter."""
    return self.retriever.get_chapter_summary(chapter, book)