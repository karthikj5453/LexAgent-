import pypdf
import os
import logging
from config import settings

logger = logging.getLogger("lexagent.chunker")

def chunk_document(file_path: str) -> list[dict]:
    """
    Reads a PDF document page-by-page, extracts text,
    and returns a list of overlapping text chunks with page references.
    Each chunk is represented as:
    {
        "id": "chunk_index",
        "text": "chunk text",
        "metadata": {
            "page_reference": "Page X",
            "document_name": "filename.pdf",
            "chunk_index": idx
        }
    }
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []
    
    chunks = []
    chunk_size = settings.chunk_size
    overlap = settings.chunk_overlap
    
    try:
        reader = pypdf.PdfReader(file_path)
        doc_name = os.path.basename(file_path)
        
        chunk_idx = 0
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if not text:
                continue
                
            # Split the page text by double newlines (paragraphs)
            paragraphs = text.split("\n\n")
            current_chunk = ""
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                # If a single paragraph is larger than chunk_size, split it into smaller slices
                if len(para) > chunk_size:
                    # Write out current pending chunk first
                    if current_chunk:
                        chunks.append({
                            "id": str(chunk_idx),
                            "text": current_chunk.strip(),
                            "metadata": {
                                "page_reference": f"Page {page_num}",
                                "document_name": doc_name,
                                "chunk_index": chunk_idx
                            }
                        })
                        chunk_idx += 1
                        current_chunk = ""
                    
                    # Split paragraph by character limit with overlap
                    start = 0
                    while start < len(para):
                        end = min(start + chunk_size, len(para))
                        chunk_text = para[start:end]
                        chunks.append({
                            "id": str(chunk_idx),
                            "text": chunk_text.strip(),
                            "metadata": {
                                "page_reference": f"Page {page_num}",
                                "document_name": doc_name,
                                "chunk_index": chunk_idx
                            }
                        })
                        chunk_idx += 1
                        start += (chunk_size - overlap)
                else:
                    # Paragraph fits. Can we merge it into the current chunk?
                    if len(current_chunk) + len(para) + 2 <= chunk_size:
                        if current_chunk:
                            current_chunk += "\n\n" + para
                        else:
                            current_chunk = para
                    else:
                        # Current chunk is full, store it
                        chunks.append({
                            "id": str(chunk_idx),
                            "text": current_chunk.strip(),
                            "metadata": {
                                "page_reference": f"Page {page_num}",
                                "document_name": doc_name,
                                "chunk_index": chunk_idx
                            }
                        })
                        chunk_idx += 1
                        
                        # Set up next chunk with overlapping text from previous
                        overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                        current_chunk = (overlap_text + "\n\n" + para).strip()
            
            # Store any remaining text on the page
            if current_chunk:
                chunks.append({
                    "id": str(chunk_idx),
                    "text": current_chunk.strip(),
                    "metadata": {
                        "page_reference": f"Page {page_num}",
                        "document_name": doc_name,
                        "chunk_index": chunk_idx
                    }
                })
                chunk_idx += 1
                
        logger.info(f"Successfully chunked PDF '{doc_name}' into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.error(f"Error during PDF chunking: {e}")
        return []
