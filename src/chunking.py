import re

# --- OFFLINE DATA PREP ---
def split_by_clause(text: str) -> list[str]:
    parts = re.split(r'(^|\s)(?=\d+\.\s)', text)
    clauses = []
    current_clause = ""
    for part in parts:
        if not part: continue
        if re.match(r'^\s*$', part) and not current_clause: continue
        if re.match(r'^\d+\.\s', part):
            if current_clause:
                clauses.append(current_clause.strip())
            current_clause = part
        else:
            current_clause += part
            
    if current_clause:
        clauses.append(current_clause.strip())
    if not clauses:
        return [text.strip()]
    return [c for c in clauses if len(c) > 5]

def split_with_overlap(text: str, max_words: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        start += (max_words - overlap)
        if len(words) - start < 25:
            remaining = " ".join(words[start:])
            if remaining and remaining not in chunks[-1]:
                chunks[-1] = (chunks[-1] + " " + remaining).strip()
            break
            
    return chunks

def chunk_document(text: str, max_words: int = 120, overlap: int = 20) -> list[str]:
    if not text:
        return []
    
    clauses = split_by_clause(text)
    all_chunks = []
    for clause in clauses:
        sub_chunks = split_with_overlap(clause, max_words=max_words, overlap=overlap)
        all_chunks.extend(sub_chunks)
    
    return all_chunks

# --- RUNTIME QUERY PROCESSING ---
def chunk_query(text: str) -> list[str]:
    if not text:
        return []
        
    max_chars = 1000
    chunks = []
    raw_chunks = text.split("\n") 
    current_chunk = ""
    
    for segment in raw_chunks:
        if len(current_chunk) + len(segment) < max_chars:
            current_chunk += segment + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = segment + "\n"
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks
