# --- START OF FILE app.py ---

import os
import PyPDF2
import nltk
import openai
from sentence_transformers import SentenceTransformer
import chromadb
from nltk.tokenize import sent_tokenize
import traceback
from supabase import create_client, Client
import datetime # For last_processed_at
import tempfile # For handling temporary files
import mimetypes # To help determine if a path is likely local (simple check)
import requests # For downloading from public URLs (Optional implementation)
from urllib.parse import urlparse # To parse URLs if needed


# --- Configuration ---
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', "YOUR_OPENAI_API_KEY")
SUPABASE_URL = os.environ.get('SUPABASE_URL', "YOUR_SUPABASE_URL")
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', "YOUR_SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_STORAGE_BUCKET = os.environ.get('SUPABASE_STORAGE_BUCKET', "YOUR_STORAGE_BUCKET_NAME")

if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
    print("Warning: OPENAI_API_KEY not set or using placeholder. OpenAI calls will fail.")
if not SUPABASE_URL or SUPABASE_URL == "YOUR_SUPABASE_URL" or \
   not SUPABASE_KEY or SUPABASE_KEY == "YOUR_SUPABASE_SERVICE_ROLE_KEY":
    print("Warning: Supabase URL or Key not set or using placeholders. Supabase operations may fail.")
if not SUPABASE_STORAGE_BUCKET or SUPABASE_STORAGE_BUCKET == "YOUR_STORAGE_BUCKET_NAME":
     print("Warning: SUPABASE_STORAGE_BUCKET name not set or using placeholder. Cannot process files from Storage.")

openai.api_key = OPENAI_API_KEY
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Supabase client initialized in app.py.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize Supabase client in app.py: {e}")
    traceback.print_exc()
    supabase = None

CHROMA_DB_PATH = os.environ.get('CHROMA_DB_PATH', "./chroma_db_store")
CHROMA_COLLECTION_NAME = "tnpsc_brain_collection"
EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'

embedding_model = None
chroma_client = None
chroma_collection = None

# --- NLTK Setup ---
try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK 'punkt' tokenizer found.")
except LookupError:
    print("NLTK 'punkt' tokenizer not found. Attempting to download...")
    try:
        nltk.download('punkt', quiet=True)
        print("'punkt' tokenizer downloaded successfully.")
    except Exception as download_error:
        print(f"Error during NLTK 'punkt' download: {download_error}")
except Exception as other_error:
     print(f"An unexpected error occurred finding NLTK 'punkt': {other_error}")


# --- Helper Functions (PDF Processing, Chunking) ---

# --- FULLY CORRECTED extract_text_from_pdf function ---
def extract_text_from_pdf(pdf_source_path, book_id="Unknown Book"):
    pages_data = []
    temp_file_path = None
    file_obj = None # Variable to hold the file object open for PyPDF2

    print(f"Attempting to extract text for book '{book_id}' from source: '{pdf_source_path}'")

    try:
        is_url = bool(urlparse(pdf_source_path).scheme)
        is_local = os.path.exists(pdf_source_path)
        # Assume if not a URL and not local, it's a storage path.
        # The order of checks in if/elif/else handles this.

        file_to_open_path = None # Initialize

        if is_local:
            print(f"Detected source as local file.")
            file_to_open_path = pdf_source_path
            file_obj = open(file_to_open_path, 'rb')

        elif is_url:
            print(f"Detected source as URL.")
            try:
                print(f"Downloading from URL: {pdf_source_path}")
                response = requests.get(pdf_source_path, stream=True)
                response.raise_for_status()
                
                content_type = response.headers.get('Content-Type', '').lower()
                if 'pdf' not in content_type and mimetypes.guess_type(pdf_source_path)[0] != 'application/pdf':
                     print(f"Warning: Downloaded content type may not be PDF: {content_type}")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    for chunk_content in response.iter_content(chunk_size=8192): # Renamed chunk to chunk_content
                        temp_file.write(chunk_content)
                    temp_file_path = temp_file.name

                file_to_open_path = temp_file_path
                print(f"Downloaded URL content and saved to temporary file: {file_to_open_path}")
                file_obj = open(file_to_open_path, 'rb')

            except Exception as url_e:
                 print(f"Error downloading PDF from URL '{pdf_source_path}' for book '{book_id}': {url_e}")
                 traceback.print_exc()
                 return None

        else: # Assume Supabase Storage path if not local and not URL
            print(f"Detected source as potential Supabase Storage path: '{pdf_source_path}'")
            if not supabase:
                 print(f"Error: Supabase client not available to download PDF '{pdf_source_path}' for book '{book_id}'.")
                 return None
            if not SUPABASE_STORAGE_BUCKET:
                 print(f"Error: SUPABASE_STORAGE_BUCKET is not configured. Cannot download '{pdf_source_path}'.")
                 return None

            print(f"Downloading from Supabase Storage bucket '{SUPABASE_STORAGE_BUCKET}': '{pdf_source_path}'")
            try:
                file_content_bytes = supabase.storage.from_(SUPABASE_STORAGE_BUCKET).download(pdf_source_path) # Renamed file_content

                if not file_content_bytes:
                    print(f"Error: Downloaded empty content for '{pdf_source_path}' from storage for book '{book_id}'.")
                    return None

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(file_content_bytes)
                    temp_file_path = temp_file.name

                file_to_open_path = temp_file_path
                print(f"Downloaded Storage content and saved to temporary file: {file_to_open_path}")
                file_obj = open(file_to_open_path, 'rb')

            except Exception as download_e:
                 print(f"Error downloading or saving PDF from Supabase Storage '{pdf_source_path}' for book '{book_id}': {download_e}")
                 traceback.print_exc()
                 return None
        
        # --- PyPDF2 Processing (Ensuring this part is complete) ---
        if file_obj: # This check ensures file_obj was successfully created/opened
            try:
                reader = PyPDF2.PdfReader(file_obj)
                if not reader.pages:
                    print(f"Warning: No pages found in PDF source: '{pdf_source_path}' for book '{book_id}'.")
                    return None # Return None if no pages

                for page_num in range(len(reader.pages)):
                    try:
                        page = reader.pages[page_num]
                        text = page.extract_text()
                        if text and text.strip():
                            pages_data.append((page_num + 1, text))
                        else:
                            print(f"Warning: No text extracted from page {page_num + 1} of source: '{pdf_source_path}' for book '{book_id}'.")
                    except Exception as page_extract_e: # More specific variable name
                        print(f"Warning: Error extracting text from page {page_num + 1} of source '{pdf_source_path}' for book '{book_id}': {page_extract_e}")
                        traceback.print_exc()
            except Exception as pypdf_e:
                 print(f"Error reading PDF with PyPDF2 from source '{pdf_source_path}' for book '{book_id}': {pypdf_e}")
                 traceback.print_exc()
                 # Ensure pages_data is None or empty to signify failure before file_obj close
                 pages_data = None # Or [] and check length later
                 return None # Return None if PyPDF2 fails
            finally:
                # Ensure the file object is closed after processing
                if file_obj and not file_obj.closed:
                     file_obj.close()
                     print(f"Closed file object for source: {pdf_source_path}")

        else: # This else corresponds to "if file_obj:" meaning file_obj was not set (e.g. download failed)
            print(f"Error: No valid file object to process for '{pdf_source_path}' for book '{book_id}'.")
            return None


    except FileNotFoundError: # This is for local file paths primarily
         print(f"Error: Local PDF file not found at {pdf_source_path} for book '{book_id}'.")
         return None
    except Exception as e:
        print(f"An unexpected error occurred processing PDF source '{pdf_source_path}' for book '{book_id}': {e}")
        traceback.print_exc()
        return None
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as cleanup_e:
                print(f"Error cleaning up temporary file {temp_file_path}: {cleanup_e}")
                traceback.print_exc()

    return pages_data if pages_data else None
# --- END OF CORRECTED extract_text_from_pdf ---


def chunk_by_sentence_window_with_source(pages_data, window_size=5, overlap=2):
    chunks_with_source = []
    if not pages_data:
        return chunks_with_source

    if 'sent_tokenize' not in dir(nltk.tokenize) or not callable(nltk.tokenize.sent_tokenize):
         print("Error: NLTK 'punkt' tokenizer is not available or callable. Cannot chunk by sentence.")
         return chunks_with_source

    for page_num, page_text in pages_data:
        if not page_text or not page_text.strip():
             continue
        try:
            sentences = sent_tokenize(page_text)
            if not sentences:
                 continue

            effective_window_size = min(window_size, len(sentences))
            if effective_window_size <= 0:
                 continue

            step_size = max(1, effective_window_size - overlap)

            for i in range(0, len(sentences) - effective_window_size + 1, step_size):
                chunk_text = " ".join(sentences[i:i + effective_window_size]) # Renamed chunk to chunk_text
                chunks_with_source.append((chunk_text, page_num))

            # Handle the last few sentences if they don't form a full window but are remaining
            # This logic might need refinement if strict windowing behavior is critical
            # The current loop `len(sentences) - effective_window_size + 1` handles most cases.
            # An explicit check for remaining sentences might be added if the last part isn't captured.
            if len(sentences) > 0 and len(sentences) < effective_window_size : # if total sentences less than window
                 chunk_text = " ".join(sentences)
                 chunks_with_source.append((chunk_text, page_num))
            # If `step_size` leads to missing the very end, this might be needed:
            # elif len(sentences) % step_size != 0 and len(sentences) > effective_window_size:
            #     # This part is a bit tricky, ensure it doesn't double-add
            #     last_start = ((len(sentences) - effective_window_size) // step_size) * step_size
            #     if last_start + step_size < len(sentences) - effective_window_size + 1: # if the loop didn't get the last possible start
            #         # This condition is complex and might be better handled by adjusting the loop range or post-loop processing
            #         pass


        except Exception as sent_error:
             print(f"Error tokenizing or chunking sentences on page {page_num}: {sent_error}")
             traceback.print_exc()
    return chunks_with_source


# --- Embedding and Storage Function ---
def generate_and_store_embeddings(book_details, collection, model):
    pdf_source_path = book_details.get('filepath')
    book_id = str(book_details.get('book_id'))
    book_title = book_details.get('title', os.path.basename(pdf_source_path) if pdf_source_path else "Unknown Title")

    if not pdf_source_path:
        print(f"Error: No filepath provided for book_id '{book_id}'. Skipping.")
        return False

    print(f"Processing book: '{book_title}' (ID: {book_id}) from source '{pdf_source_path}' for collection '{collection.name}'")

    # CORRECTED: extract_text_from_pdf determines source type internally
    pages_data = extract_text_from_pdf(pdf_source_path, book_id=book_id)

    if not pages_data:
        print(f"Failed to extract text from '{pdf_source_path}' for book_id '{book_id}'. No embeddings will be generated.")
        return False

    if 'sent_tokenize' not in dir(nltk.tokenize) or not callable(nltk.tokenize.sent_tokenize):
        print("Critical Error: NLTK 'punkt' tokenizer is not available. Cannot chunk text.")
        return False

    chunks_with_source = chunk_by_sentence_window_with_source(pages_data)
    if not chunks_with_source:
        print(f"No valid text chunks found for '{pdf_source_path}'. No embeddings will be generated for book_id '{book_id}'.")
        return False

    print(f"Generated {len(chunks_with_source)} chunks from '{book_title}'. Now creating embeddings...")
    texts_to_embed = [text for text, _ in chunks_with_source]
    try:
        embeddings = model.encode(texts_to_embed, show_progress_bar=True)
    except Exception as e:
        print(f"Error generating embeddings for book_id '{book_id}': {e}")
        traceback.print_exc()
        return False

    documents = [text for text, _ in chunks_with_source]
    page_numbers_for_chunks = [str(page) for _, page in chunks_with_source]
    ids = [f"{book_id}_page_{page_numbers_for_chunks[i]}_chunk_{i}" for i in range(len(documents))]
    metadatas = [{
        "page": page_numbers_for_chunks[i],
        "source_document_id": book_id,
        "book_title": book_title,
        "original_source_path": pdf_source_path,
        "original_filename": os.path.basename(pdf_source_path)
    } for i in range(len(documents))]

    batch_size = 100
    num_batches = (len(documents) + batch_size - 1) // batch_size
    print(f"Adding {len(documents)} chunks in {num_batches} batches to Chroma DB...")

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(documents))
        batch_embeddings = embeddings[start_idx:end_idx]
        batch_documents = documents[start_idx:end_idx]
        batch_metadatas = metadatas[start_idx:end_idx]
        batch_ids = ids[start_idx:end_idx]

        try:
            collection.add(
                embeddings=batch_embeddings.tolist(),
                documents=batch_documents,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            print(f"Successfully added batch {i+1}/{num_batches} ({len(batch_documents)} chunks) from '{book_title}' to Chroma DB.")
        except Exception as e:
            print(f"Error adding batch {i+1} of embeddings for book_id '{book_id}' to Chroma: {e}")
            traceback.print_exc()
            # Optionally return False here if a batch failure is critical
    print(f"Completed adding chunks for '{book_title}' (Book ID: {book_id}).")
    return True


# --- RAG Core Functions ---
def find_relevant_chunks_chroma(query_text, collection, model, top_n=5, similarity_threshold=0.5, where_filter=None):
    if not collection:
        print("Error: Chroma collection is not available for querying.")
        return []
    if not model:
         print("Error: Embedding model is not loaded for querying.")
         return []
    try:
        query_embedding = model.encode([query_text]).tolist()
        query_params = {
            "query_embeddings": query_embedding,
            "n_results": top_n,
            "include": ["documents", "metadatas", "distances"]
        }
        if where_filter:
            query_params["where"] = where_filter
            print(f"Chroma query with filter: {where_filter}")
        results = collection.query(**query_params)
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        traceback.print_exc()
        return []

    relevant_chunks_info = []
    if results and results.get('documents') and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            distance = results['distances'][0][i]
            similarity_score = 1 - distance # Assuming 'cosine' space
            if similarity_score >= similarity_threshold:
                metadata = results['metadatas'][0][i]
                relevant_chunks_info.append({
                    'chunk_text': results['documents'][0][i],
                    'page_number': metadata.get('page', 'N/A'),
                    'source_document_id': metadata.get('source_document_id', 'N/A'),
                    'book_title': metadata.get('book_title', 'N/A'),
                    'similarity': round(similarity_score, 4)
                })
    return sorted(relevant_chunks_info, key=lambda x: x['similarity'], reverse=True)

def get_answer_from_openai(question, context_chunks):
    if not openai.api_key or openai.api_key == "YOUR_OPENAI_API_KEY":
        print("Error: OpenAI API key not configured. Cannot generate answer.")
        return "Error: AI service is not configured."
    context_str = "\n\n---\n\n".join([chunk['chunk_text'] for chunk in context_chunks]) if context_chunks else "No specific relevant context found in the documents."
    system_instruction = (
        "You are a helpful AI assistant specializing in TNPSC exam preparation materials. "
        "Your task is to answer the user's question accurately based *only* on the provided document context. "
        "Do not use outside knowledge. If the answer cannot be found in the context, state clearly that the information is not available in the provided documents. "
        "Provide concise and direct answers."
    )
    user_prompt = f"Context from documents:\n\"\"\"\n{context_str}\n\"\"\"\n\nQuestion: {question}\n\nAnswer:"
    try:
        response = openai.chat.completions.create( # Corrected attribute access for openai > v1.0
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        generated_text = response.choices[0].message.content.strip()
        return generated_text
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        traceback.print_exc()
        return f"Error generating answer from AI: {e}"


# --- System Initialization ---
def initialize_system():
    global embedding_model, chroma_client, chroma_collection, supabase
    print("--- System Initialization Start ---")

    if supabase is None:
        print("CRITICAL ERROR: Supabase client failed to initialize earlier. Cannot proceed with system initialization.")
        return False

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}")
        traceback.print_exc()
        return False

    print(f"Initializing ChromaDB client at path: {CHROMA_DB_PATH}...")
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        print("ChromaDB client initialized.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize ChromaDB client at '{CHROMA_DB_PATH}': {e}")
        traceback.print_exc()
        return False

    print(f"Getting or creating ChromaDB collection: {CHROMA_COLLECTION_NAME}...")
    try:
        chroma_collection = chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Chroma collection '{CHROMA_COLLECTION_NAME}' is ready (Current item count: {chroma_collection.count()}).")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to setup ChromaDB collection '{CHROMA_COLLECTION_NAME}': {e}")
        traceback.print_exc()
        return False

    if not SUPABASE_STORAGE_BUCKET or SUPABASE_STORAGE_BUCKET == "YOUR_STORAGE_BUCKET_NAME":
         print("Warning: SUPABASE_STORAGE_BUCKET name is not configured. Skipping book processing from Storage during initialization.")
         # Continue initialization even if bucket isn't set, but log it.
         # Reprocessing endpoint can still be used later if bucket is configured.

    print("Fetching book list from Supabase database for initial processing...")
    try:
        response = supabase.table('books')\
                           .select('book_id, title, filepath, processed, subject, author, version, last_processed_at')\
                           .execute()

        if hasattr(response, 'error') and response.error:
            print(f"Error fetching books from Supabase: {response.error.message if response.error else 'Unknown error'}")
            traceback.print_exc()
            # Decide if this is critical. For now, we'll let system initialize but log error.
        elif response.data is not None:
            books_list = response.data
            print(f"Found {len(books_list)} books in Supabase.")
            processed_count = 0
            skipped_count = 0
            failed_processing_count = 0

            for book_details in books_list:
                book_id_str = str(book_details.get('book_id'))
                book_title = book_details.get('title', "Untitled Book")
                book_filepath = book_details.get('filepath')
                needs_processing = not book_details.get('processed', False)

                if not book_filepath:
                     print(f"Skipping book '{book_title}' (ID: {book_id_str}): No filepath provided in DB.")
                     skipped_count += 1
                     continue
                
                # Only attempt to process if storage bucket is configured
                if not SUPABASE_STORAGE_BUCKET or SUPABASE_STORAGE_BUCKET == "YOUR_STORAGE_BUCKET_NAME":
                    if needs_processing and not urlparse(book_filepath).scheme and not os.path.exists(book_filepath): # if it looks like a storage path
                        print(f"Skipping processing for '{book_title}' (ID: {book_id_str}): SUPABASE_STORAGE_BUCKET not configured and filepath seems to be a storage path.")
                        skipped_count +=1
                        continue


                if needs_processing:
                    print(f"\n--- Queuing Book for Processing: '{book_title}' (ID: {book_id_str}) ---")
                    print(f"  Source filepath: '{book_filepath}'")
                    if generate_and_store_embeddings(book_details, chroma_collection, embedding_model):
                        print(f"Successfully processed and stored embeddings for: '{book_title}'")
                        try:
                            update_response = supabase.table('books').update({ # Store response
                                'processed': True,
                                'last_processed_at': datetime.datetime.utcnow().isoformat()
                            }).eq('book_id', book_id_str).execute()
                            if hasattr(update_response, 'error') and update_response.error:
                                print(f"Error updating Supabase 'processed' status for book '{book_id_str}': {update_response.error.message if update_response.error else 'Unknown DB error'}")
                            else:
                                print(f"Updated 'processed' status for book '{book_id_str}' in Supabase.")
                            processed_count += 1
                        except Exception as e_supa_update: # Renamed e_supa
                            print(f"Exception updating Supabase 'processed' status for book '{book_id_str}': {e_supa_update}")
                            traceback.print_exc()
                    else:
                        print(f"Failed to process book: '{book_title}' (ID: {book_id_str}). Embeddings were NOT generated or stored.")
                        failed_processing_count += 1
                else:
                    print(f"Book '{book_title}' (ID: {book_id_str}) already marked as processed. Skipping initial load.")
                    skipped_count += 1

            print(f"\n--- Initial Book Processing Summary ---")
            print(f"Total books in DB: {len(books_list)}")
            print(f"Processed during this run: {processed_count}")
            print(f"Skipped (already processed, no filepath, or storage not configured): {skipped_count}")
            print(f"Failed processing attempts: {failed_processing_count}")
        else:
            print("Supabase response for books table did not contain data (response.data is None).")
    except Exception as e_fetch_books: # Renamed e
        print(f"Error during Supabase book fetching or initial processing loop: {e_fetch_books}")
        traceback.print_exc()

    print("--- System Initialization End ---")
    return embedding_model is not None and chroma_client is not None and chroma_collection is not None and supabase is not None

# --- END OF FILE app.py ---