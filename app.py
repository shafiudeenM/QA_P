# --- START OF FILE app.py ---

import os
import PyPDF2
import nltk
import openai
from sentence_transformers import SentenceTransformer
# import chromadb # Removed ChromaDB
from nltk.tokenize import sent_tokenize # Make sure this is imported
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

EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'
EMBEDDING_DIMENSION = 768 # Standard for all-mpnet-base-v2

SUPABASE_EMBEDDINGS_TABLE_NAME = "embeddings"
SUPABASE_RPC_MATCH_EMBEDDINGS = "match_embeddings" # Ensure this RPC function exists in Supabase

embedding_model = None

# --- NLTK Setup (REVISED AND MORE ROBUST) ---
NLTK_PUNKT_RESOURCE_PATH = 'tokenizers/punkt'
NLTK_PUNKT_DOWNLOAD_ID = 'punkt'
NLTK_PUNKT_TAB_RESOURCE_PATH = 'tokenizers/punkt_tab' # Path for punkt_tab
NLTK_PUNKT_TAB_DOWNLOAD_ID = 'punkt_tab'            # ID for punkt_tab

def ensure_nltk_resource(resource_path, download_id):
    """
    Ensures a specific NLTK resource is available.
    Downloads it if necessary. Returns True if successful/available, False otherwise.
    """
    try:
        nltk.data.find(resource_path)
        print(f"NLTK '{download_id}' resource found ({resource_path}).")
        return True
    except LookupError:
        print(f"NLTK '{download_id}' resource not found ({resource_path}). Attempting to download...")
        try:
            nltk.download(download_id, quiet=False)
            print(f"NLTK '{download_id}' download initiated.")
            # Verify after attempting download
            nltk.data.find(resource_path)
            print(f"Successfully verified NLTK '{download_id}' resource ({resource_path}) after download attempt.")
            return True
        except Exception as download_error:
            print(f"CRITICAL NLTK ERROR: Failed to download or verify NLTK '{download_id}' resource ({resource_path}): {download_error}")
            print("Details of download error:")
            traceback.print_exc()
            print(f"Search paths were: {nltk.data.path}")
            print(f"Sentence tokenization requiring '{download_id}' will FAIL. Ensure NLTK data can be downloaded or is manually placed.")
            return False
    except Exception as e:
        print(f"An unexpected error occurred while checking for NLTK '{download_id}' ({resource_path}): {e}")
        traceback.print_exc()
        return False

# Call this at module level to ensure NLTK resources run on import
NLTK_PUNKT_IS_AVAILABLE = ensure_nltk_resource(NLTK_PUNKT_RESOURCE_PATH, NLTK_PUNKT_DOWNLOAD_ID)
NLTK_PUNKT_TAB_IS_AVAILABLE = ensure_nltk_resource(NLTK_PUNKT_TAB_RESOURCE_PATH, NLTK_PUNKT_TAB_DOWNLOAD_ID)

ALL_REQUIRED_NLTK_AVAILABLE = NLTK_PUNKT_IS_AVAILABLE and NLTK_PUNKT_TAB_IS_AVAILABLE


# --- Helper Functions (PDF Processing, Chunking) ---
def extract_text_from_pdf(pdf_source_path, book_id="Unknown Book"):
    pages_data = []
    temp_file_path = None
    file_obj = None

    print(f"Attempting to extract text for book '{book_id}' from source: '{pdf_source_path}'")

    try:
        is_url = bool(urlparse(pdf_source_path).scheme)
        is_local = os.path.exists(pdf_source_path)
        file_to_open_path = None

        if is_local:
            print(f"Detected source as local file: {pdf_source_path}")
            file_to_open_path = pdf_source_path
            file_obj = open(file_to_open_path, 'rb')
        elif is_url:
            print(f"Detected source as URL: {pdf_source_path}")
            try:
                print(f"Downloading from URL: {pdf_source_path}")
                response = requests.get(pdf_source_path, stream=True, timeout=30) # Added timeout
                response.raise_for_status()
                content_type = response.headers.get('Content-Type', '').lower()
                if 'pdf' not in content_type and mimetypes.guess_type(pdf_source_path)[0] != 'application/pdf':
                     print(f"Warning: Downloaded content type may not be PDF: {content_type} for URL {pdf_source_path}")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    for chunk_content in response.iter_content(chunk_size=8192):
                        temp_file.write(chunk_content)
                    temp_file_path = temp_file.name
                file_to_open_path = temp_file_path
                print(f"Downloaded URL content and saved to temporary file: {file_to_open_path}")
                file_obj = open(file_to_open_path, 'rb')
            except requests.exceptions.RequestException as url_e: # More specific exception for requests
                 print(f"Error downloading PDF from URL '{pdf_source_path}' for book '{book_id}': {url_e}")
                 traceback.print_exc()
                 return None
            except Exception as e: # Catch other potential errors during URL processing
                 print(f"An unexpected error occurred while processing URL '{pdf_source_path}' for book '{book_id}': {e}")
                 traceback.print_exc()
                 return None
        else: # Assumed to be a Supabase Storage path
            print(f"Detected source as potential Supabase Storage path: '{pdf_source_path}'")
            if not supabase:
                 print(f"Error: Supabase client not available to download PDF '{pdf_source_path}' for book '{book_id}'.")
                 return None
            if not SUPABASE_STORAGE_BUCKET:
                 print(f"Error: SUPABASE_STORAGE_BUCKET is not configured. Cannot download '{pdf_source_path}'.")
                 return None
            print(f"Downloading from Supabase Storage bucket '{SUPABASE_STORAGE_BUCKET}': '{pdf_source_path}'")
            try:
                file_content_bytes = supabase.storage.from_(SUPABASE_STORAGE_BUCKET).download(pdf_source_path)
                if not file_content_bytes:
                    print(f"Error: Downloaded empty content for '{pdf_source_path}' from Supabase Storage for book '{book_id}'.")
                    return None
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(file_content_bytes)
                    temp_file_path = temp_file.name
                file_to_open_path = temp_file_path
                print(f"Downloaded Supabase Storage content and saved to temporary file: {file_to_open_path}")
                file_obj = open(file_to_open_path, 'rb')
            except Exception as download_e:
                 print(f"Error downloading or saving PDF from Supabase Storage '{pdf_source_path}' for book '{book_id}': {download_e}")
                 traceback.print_exc()
                 return None

        if file_obj:
            try:
                reader = PyPDF2.PdfReader(file_obj)
                if not reader.pages:
                    print(f"Warning: No pages found in PDF source: '{pdf_source_path}' for book '{book_id}'.")
                    return None # Return None if no pages
                for page_num in range(len(reader.pages)):
                    try:
                        page = reader.pages[page_num]
                        text = page.extract_text()
                        if text and text.strip(): # Ensure text is not just whitespace
                            pages_data.append((page_num + 1, text))
                        # else: # Commented out to reduce noise for empty pages
                        #     print(f"Warning: No text extracted from page {page_num + 1} of source: '{pdf_source_path}' for book '{book_id}'.")
                    except Exception as page_extract_e:
                        print(f"Warning: Error extracting text from page {page_num + 1} of source '{pdf_source_path}' for book '{book_id}': {page_extract_e}")
                        # traceback.print_exc() # Can be noisy
            except PyPDF2.errors.PdfReadError as pypdf_read_e: # More specific PyPDF2 error
                 print(f"PyPDF2 Read Error for PDF from source '{pdf_source_path}' for book '{book_id}': {pypdf_read_e}. File might be corrupted or not a valid PDF.")
                 traceback.print_exc()
                 return None
            except Exception as pypdf_e:
                 print(f"Error reading PDF with PyPDF2 from source '{pdf_source_path}' for book '{book_id}': {pypdf_e}")
                 traceback.print_exc()
                 return None # Return None on error
            finally:
                if file_obj and not file_obj.closed:
                     file_obj.close()
                     print(f"Closed file object for source: {pdf_source_path}")
        else:
            print(f"Error: No valid file object to process for '{pdf_source_path}' for book '{book_id}'.")
            return None # Return None if no file object
    except FileNotFoundError:
         print(f"Error: Local PDF file not found at '{pdf_source_path}' for book '{book_id}'.")
         return None
    except Exception as e:
        print(f"An unexpected error occurred processing PDF source '{pdf_source_path}' for book '{book_id}': {e}")
        traceback.print_exc()
        return None # Return None on any other unexpected error
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as cleanup_e:
                print(f"Error cleaning up temporary file {temp_file_path}: {cleanup_e}")
                traceback.print_exc()
    
    return pages_data if pages_data else None


def chunk_by_sentence_window_with_source(pages_data, window_size=5, overlap=2):
    chunks_with_source = []
    if not pages_data:
        return chunks_with_source

    if not ALL_REQUIRED_NLTK_AVAILABLE: # Check the global flag for all NLTK resources
         print("CRITICAL NLTK ERROR: Required 'punkt' and/or 'punkt_tab' tokenizers are not available. Cannot chunk by sentence.")
         return chunks_with_source

    for page_num, page_text in pages_data:
        if not page_text or not page_text.strip():
             continue
        try:
            sentences = sent_tokenize(page_text)
            if not sentences:
                 continue

            num_sentences = len(sentences)
            if num_sentences == 0:
                continue

            # Determine step size, ensuring it's at least 1
            # If window_size is 1 and overlap is 0 (or more), step should be 1.
            # If window_size is 5 and overlap is 2, step is 3.
            step = max(1, window_size - overlap)

            for i in range(0, num_sentences, step):
                # The end of the slice will be min(i + window_size, num_sentences)
                # to handle the tail end of sentences correctly.
                chunk_sentence_list = sentences[i : i + window_size]
                if chunk_sentence_list: # Ensure the list is not empty
                    chunk_text = " ".join(chunk_sentence_list)
                    chunks_with_source.append((chunk_text, page_num))
                # The loop structure `range(0, num_sentences, step)` and slicing `sentences[i : i + window_size]`
                # naturally handles all sentences. The last slice `sentences[last_i : last_i + window_size]`
                # will correctly become `sentences[last_i : num_sentences]` if `last_i + window_size` exceeds `num_sentences`.

        except LookupError as le:
            print(f"NLTK LookupError during sentence tokenization on page {page_num}: {le}")
            print("This indicates a required NLTK resource (e.g., 'punkt' or 'punkt_tab') might still be missing or inaccessible despite download attempts.")
            print("Further processing of this page's text will be skipped.")
            continue
        except Exception as sent_error:
             print(f"Error tokenizing or chunking sentences on page {page_num}: {sent_error}")
             traceback.print_exc()
    return chunks_with_source


# --- Embedding and Storage Function (Modified for Supabase pgvector) ---
def generate_and_store_embeddings(book_details, model):
    pdf_source_path = book_details.get('filepath')
    book_id = str(book_details.get('book_id')) # Ensure book_id is a string
    book_title = book_details.get('title', os.path.basename(pdf_source_path) if pdf_source_path else "Unknown Title")

    if not pdf_source_path:
        print(f"Error: No filepath provided for book_id '{book_id}'. Skipping.")
        return False

    if not supabase:
        print(f"Error: Supabase client not available for storing embeddings for book_id '{book_id}'.")
        return False

    print(f"Processing book: '{book_title}' (ID: {book_id}) from source '{pdf_source_path}' for Supabase table '{SUPABASE_EMBEDDINGS_TABLE_NAME}'")

    pages_data = extract_text_from_pdf(pdf_source_path, book_id=book_id)

    if not pages_data:
        print(f"Failed to extract text from '{pdf_source_path}' for book_id '{book_id}'. No embeddings will be generated.")
        return False

    if not ALL_REQUIRED_NLTK_AVAILABLE: # Check the global flag before calling chunking
        print(f"CRITICAL NLTK ERROR: Required NLTK tokenizers not available for book '{book_title}'. Skipping embedding generation.")
        return False

    chunks_with_source = chunk_by_sentence_window_with_source(pages_data) # Default window_size=5, overlap=2
    if not chunks_with_source:
        print(f"No valid text chunks found for '{pdf_source_path}'. No embeddings will be generated for book_id '{book_id}'.")
        return False

    print(f"Generated {len(chunks_with_source)} chunks from '{book_title}'. Now creating embeddings...")
    texts_to_embed = [text for text, _ in chunks_with_source]
    try:
        embeddings_vectors = model.encode(texts_to_embed, show_progress_bar=True)
    except Exception as e:
        print(f"Error generating embeddings for book_id '{book_id}': {e}")
        traceback.print_exc()
        return False

    records_to_insert = []
    for i, (chunk_text, page_num) in enumerate(chunks_with_source):
        record = {
            "book_id": book_id, # Should be string
            "page_number": int(page_num),
            "chunk_text": chunk_text,
            "embedding": embeddings_vectors[i].tolist(), # pgvector expects list of floats
            "metadata": { # Store as JSONB
                "book_title": book_title,
                "original_source_path": pdf_source_path, # For reference
                "original_filename": os.path.basename(pdf_source_path) if pdf_source_path else None
            }
        }
        records_to_insert.append(record)

    batch_size = 100 # Adjust as needed based on performance and Supabase limits
    num_batches = (len(records_to_insert) + batch_size - 1) // batch_size
    print(f"Adding {len(records_to_insert)} embeddings in {num_batches} batches to Supabase table '{SUPABASE_EMBEDDINGS_TABLE_NAME}' for book_id '{book_id}'...")

    all_successful = True
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(records_to_insert))
        batch_records = records_to_insert[start_idx:end_idx]

        try:
            response = supabase.table(SUPABASE_EMBEDDINGS_TABLE_NAME).insert(batch_records).execute()
            if hasattr(response, 'error') and response.error:
                print(f"Error adding batch {i+1}/{num_batches} of embeddings for book_id '{book_id}' to Supabase: {response.error.message if response.error else 'Unknown DB error'}")
                # traceback.print_exc() # Can be very verbose for batch errors
                all_successful = False # Mark as not all successful if any batch fails
            elif response.data: # supabase-py v2 returns data on success
                print(f"Successfully added batch {i+1}/{num_batches} ({len(response.data)} embeddings confirmed) from '{book_title}' to Supabase.")
            else: # Should not happen if no error and data is expected
                 print(f"Batch {i+1}/{num_batches} for '{book_title}' to Supabase: No data in response but no explicit error. Check DB.")
        except Exception as e:
            print(f"Exception during Supabase insert for batch {i+1} of book_id '{book_id}': {e}")
            traceback.print_exc()
            all_successful = False

    if all_successful:
        print(f"Completed adding all embeddings for '{book_title}' (Book ID: {book_id}).")
    else:
        print(f"Completed adding embeddings for '{book_title}' (Book ID: {book_id}) with some errors during DB insertion.")
    return all_successful


# --- System Initialization ---
def initialize_system():
    global embedding_model, supabase
    print("--- System Initialization Start ---")

    if not ALL_REQUIRED_NLTK_AVAILABLE:
        print("CRITICAL WARNING: Not all required NLTK resources ('punkt' and/or 'punkt_tab') are available. Chunking functionality will be impaired or fail.")
        # Depending on strictness, you might return False here.
        # For now, let's allow system to try to initialize other parts, but it's a critical state.

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
        return False # Cannot proceed without embedding model

    print(f"Checking Supabase embeddings table: '{SUPABASE_EMBEDDINGS_TABLE_NAME}'...")
    try:
        # Test table accessibility and get a count
        count_response = supabase.table(SUPABASE_EMBEDDINGS_TABLE_NAME).select('id', count='exact').limit(0).execute()
        if hasattr(count_response, 'error') and count_response.error:
            print(f"Warning: Could not get count from '{SUPABASE_EMBEDDINGS_TABLE_NAME}'. Table might not exist or be accessible: {count_response.error.message if count_response.error else 'Unknown error'}")
            print(f"Ensure the table '{SUPABASE_EMBEDDINGS_TABLE_NAME}' and RPC function '{SUPABASE_RPC_MATCH_EMBEDDINGS}' are created in Supabase as per documentation.")
        elif count_response.count is not None:
            print(f"Supabase table '{SUPABASE_EMBEDDINGS_TABLE_NAME}' seems accessible (Current item count: {count_response.count}).")
        else:
             print(f"Supabase table '{SUPABASE_EMBEDDINGS_TABLE_NAME}' count returned None. Ensure it's set up correctly.")

    except Exception as e:
        print(f"CRITICAL ERROR: Failed to interact with Supabase table '{SUPABASE_EMBEDDINGS_TABLE_NAME}': {e}")
        print("Ensure the table is created in Supabase, pgvector extension is enabled, and network is reachable.")
        traceback.print_exc()
        # Potentially return False here if table access is critical for startup,
        # but initial book processing might still run if table is created later by user.

    if not SUPABASE_STORAGE_BUCKET or SUPABASE_STORAGE_BUCKET == "YOUR_STORAGE_BUCKET_NAME":
         print("Warning: SUPABASE_STORAGE_BUCKET name is not configured. Skipping book processing from Supabase Storage during initialization.")

    print("Fetching book list from Supabase 'books' table for initial processing...")
    try:
        # Select only necessary fields for processing logic
        books_response = supabase.table('books')\
                           .select('book_id, title, filepath, processed, subject, author, version, last_processed_at')\
                           .execute()

        if hasattr(books_response, 'error') and books_response.error:
            print(f"Error fetching books from Supabase 'books' table: {books_response.error.message if books_response.error else 'Unknown error'}")
            traceback.print_exc()
        elif books_response.data is not None:
            books_list = books_response.data
            print(f"Found {len(books_list)} books in Supabase 'books' table.")
            processed_during_run_count = 0
            skipped_count = 0
            failed_processing_count = 0

            for book_details in books_list:
                book_id_str = str(book_details.get('book_id')) # Ensure string
                book_title = book_details.get('title', "Untitled Book")
                book_filepath = book_details.get('filepath')
                needs_processing = not book_details.get('processed', False) # Process if 'processed' is False or missing

                if not book_filepath:
                     print(f"Skipping book '{book_title}' (ID: {book_id_str}): No filepath provided in DB.")
                     skipped_count += 1
                     continue

                # Check if storage bucket is needed and configured
                is_storage_path = not urlparse(book_filepath).scheme and not os.path.exists(book_filepath)
                if is_storage_path and (not SUPABASE_STORAGE_BUCKET or SUPABASE_STORAGE_BUCKET == "YOUR_STORAGE_BUCKET_NAME"):
                    if needs_processing:
                        print(f"Skipping processing for '{book_title}' (ID: {book_id_str}): SUPABASE_STORAGE_BUCKET not configured and filepath '{book_filepath}' appears to be a storage path.")
                        skipped_count +=1
                    continue # Skip if storage path but bucket not configured

                if needs_processing:
                    if not ALL_REQUIRED_NLTK_AVAILABLE:
                        print(f"Skipping processing of book '{book_title}' (ID: {book_id_str}) because required NLTK resources are unavailable.")
                        failed_processing_count += 1
                        continue

                    print(f"\n--- Queuing Book for Processing: '{book_title}' (ID: {book_id_str}) ---")
                    print(f"  Source filepath: '{book_filepath}'")
                    if generate_and_store_embeddings(book_details, embedding_model):
                        print(f"Successfully processed and stored embeddings for: '{book_title}'")
                        try:
                            update_response = supabase.table('books').update({
                                'processed': True,
                                'last_processed_at': datetime.datetime.now(datetime.timezone.utc).isoformat() # Use timezone-aware UTC
                            }).eq('book_id', book_id_str).execute()

                            if hasattr(update_response, 'error') and update_response.error:
                                print(f"Error updating Supabase 'processed' status for book '{book_id_str}': {update_response.error.message if update_response.error else 'Unknown DB error'}")
                            # elif update_response.data: # Check if data indicates success
                            #     print(f"Updated 'processed' status for book '{book_id_str}' in Supabase.")
                            else: # No error, assume success or check response structure if needed
                                print(f"Updated 'processed' status for book '{book_id_str}' in Supabase.")
                            processed_during_run_count += 1
                        except Exception as e_supa_update:
                            print(f"Exception updating Supabase 'processed' status for book '{book_id_str}': {e_supa_update}")
                            traceback.print_exc()
                    else:
                        print(f"Failed to process book: '{book_title}' (ID: {book_id_str}). Embeddings were NOT generated or stored completely.")
                        failed_processing_count += 1
                else:
                    print(f"Book '{book_title}' (ID: {book_id_str}) already marked as processed. Skipping initial load.")
                    skipped_count += 1

            print(f"\n--- Initial Book Processing Summary ---")
            print(f"Total books in DB: {len(books_list)}")
            print(f"Processed during this run: {processed_during_run_count}")
            print(f"Skipped (already processed, no filepath, or storage not configured for path): {skipped_count}")
            print(f"Failed processing attempts (incl. NLTK issues): {failed_processing_count}")
        else: # books_response.data is None
            print("Supabase response for 'books' table did not contain data. Table might be empty or an issue occurred.")
    except Exception as e_fetch_books:
        print(f"Error during Supabase book fetching or initial processing loop: {e_fetch_books}")
        traceback.print_exc()

    print("--- System Initialization End ---")
    # System is considered initialized if core components are up, even if NLTK has issues (it will warn).
    # However, if NLTK is absolutely critical for *all* operations, then include ALL_REQUIRED_NLTK_AVAILABLE.
    return embedding_model is not None and supabase is not None and ALL_REQUIRED_NLTK_AVAILABLE


# --- Example Functions for RAG (to be completed based on your Supabase pgvector setup) ---

def find_relevant_chunks_supabase(query_text, model, top_n=5, similarity_threshold=0.5, filter_book_ids=None):
    """
    Finds relevant chunks from Supabase pgvector based on semantic similarity.
    Placeholder - requires implementation of Supabase RPC call or direct query.
    """
    if not supabase or not model:
        print("Error: Supabase client or embedding model not available for searching.")
        return []

    try:
        query_embedding = model.encode(query_text).tolist()
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        traceback.print_exc()
        return []

    params = {
        'query_embedding': query_embedding,
        'similarity_threshold': float(similarity_threshold), # Ensure float
        'match_count': int(top_n) # Ensure int
    }
    if filter_book_ids and isinstance(filter_book_ids, list) and len(filter_book_ids) > 0:
        # Ensure book_ids are strings if your DB expects that (recommended)
        params['filter_book_ids'] = [str(bid) for bid in filter_book_ids]
        rpc_to_call = "match_embeddings_filtered" # Example: you might need a separate RPC for filtering
        print(f"Calling RPC '{rpc_to_call}' with book_id filter: {filter_book_ids}")
    else:
        rpc_to_call = SUPABASE_RPC_MATCH_EMBEDDINGS # Default RPC without book_id filter
        print(f"Calling RPC '{rpc_to_call}' without book_id filter.")


    try:
        print(f"Executing Supabase RPC '{rpc_to_call}' with top_n={top_n}, threshold={similarity_threshold}")
        response = supabase.rpc(rpc_to_call, params).execute()

        if hasattr(response, 'error') and response.error:
            print(f"Error calling Supabase RPC '{rpc_to_call}': {response.error.message if response.error else 'Unknown RPC error'}")
            traceback.print_exc()
            return []
        
        if response.data:
            print(f"Retrieved {len(response.data)} relevant chunks from Supabase.")
            # Transform data if necessary to match expected format for get_answer_from_openai
            # Expected format: list of dicts, each with 'chunk_text', 'page_number', 'book_title', 'source_document_id' (book_id), 'similarity'
            relevant_chunks_transformed = []
            for item in response.data:
                # Assuming your RPC returns fields like: id, book_id, page_number, chunk_text, metadata (jsonb), similarity_score
                metadata = item.get('metadata', {})
                chunk_info = {
                    'chunk_text': item.get('chunk_text'),
                    'page_number': item.get('page_number'),
                    'book_title': metadata.get('book_title', item.get('book_id')), # Fallback for title
                    'source_document_id': str(item.get('book_id')), # Ensure book_id is string
                    'similarity': item.get('similarity') or item.get('similarity_score') # Adjust based on your RPC output field name
                }
                # Filter out any chunks that are missing critical information
                if chunk_info['chunk_text'] and chunk_info['source_document_id']:
                    relevant_chunks_transformed.append(chunk_info)
            return relevant_chunks_transformed
        else:
            print("No relevant chunks found or RPC returned empty data.")
            return []

    except Exception as e:
        print(f"Exception during Supabase RPC call '{rpc_to_call}': {e}")
        traceback.print_exc()
        return []


def get_answer_from_openai(question, relevant_chunks, model_name="gpt-3.5-turbo"):
    """
    Generates an answer using OpenAI's API based on the question and relevant chunks.
    """
    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
        print("OpenAI API key not configured. Cannot generate answer.")
        return "OpenAI API key not configured. I am unable to answer at this time."

    if not relevant_chunks:
        # Fallback or different prompt if no relevant chunks are found
        # For now, just indicate no context was found.
        # You might choose to still ask OpenAI the question without context,
        # or return a specific message.
        # return "I couldn't find specific information in the documents to answer your question. You can try rephrasing or asking a more general question."
        
        # Try asking OpenAI directly without context if no chunks found
        print("No relevant chunks found from documents. Asking OpenAI directly.")
        context_str = "No specific context found in the provided documents."

    else:
        context_str = "\n\n---\n\n".join([
            f"From Book: '{chunk.get('book_title', 'Unknown Title')}', Page: {chunk.get('page_number', 'N/A')}\nContent: {chunk['chunk_text']}"
            for chunk in relevant_chunks
        ])

    prompt = f"""
You are an AI assistant for TNPSC (Tamil Nadu Public Service Commission) exam preparation.
Based on the following context from TNPSC study materials, answer the question.
If the answer is not found in the context, state that clearly. Do not make up information.
Be concise and focus on the information relevant to the question.

Context:
{context_str}

Question: {question}

Answer:
"""
    print(f"\n--- OpenAI Prompt for question: '{question[:50]}...' ---")
    # print(prompt) # Can be very long, print only if debugging
    print(f"Context length: {len(context_str)} characters")
    print(f"Using OpenAI model: {model_name}")

    try:
        response = openai.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant for TNPSC exam preparation. Answer based on the provided context. If the context doesn't contain the answer, say so."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, # Lower temperature for more factual, less creative answers
        )
        ai_answer = response.choices[0].message.content.strip()
        print(f"OpenAI generated answer: {ai_answer[:100]}...")
        return ai_answer
    except openai.APIError as e: # More specific OpenAI errors
        print(f"OpenAI API Error: {e}")
        traceback.print_exc()
        error_message = f"An error occurred with the OpenAI API: {e}. Please try again later."
        if hasattr(e, 'status_code') and e.status_code == 401:
            error_message = "OpenAI API Error: Authentication failed. Please check your API key."
        elif hasattr(e, 'status_code') and e.status_code == 429:
            error_message = "OpenAI API Error: Rate limit exceeded or quota reached. Please check your OpenAI plan and usage."
        return error_message
    except Exception as e:
        print(f"Error getting answer from OpenAI: {e}")
        traceback.print_exc()
        return f"An error occurred while trying to generate an answer: {str(e)}"

# --- END OF FILE app.py ---