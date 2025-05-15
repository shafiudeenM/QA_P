import os
import PyPDF2
import nltk
import openai
from sentence_transformers import SentenceTransformer
import chromadb
# Removed Flask imports - the Flask app instance and routes are now in fi.py
# from flask import Flask, request, jsonify
# from flask_swagger_ui import get_swaggerui_blueprint
from nltk.tokenize import sent_tokenize
import traceback
from supabase import create_client, Client
import datetime # For last_processed_at
import tempfile # For handling temporary files
import mimetypes # To help determine if a path is likely local (simple check)
import requests # For downloading from public URLs (Optional implementation)
from urllib.parse import urlparse # To parse URLs if needed


# --- Configuration ---
# IMPORTANT: For deployment, read these from Environment Variables!
# Example: os.environ.get('OPENAI_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', "YOUR_OPENAI_API_KEY") # Replace placeholder or use env var
SUPABASE_URL = os.environ.get('SUPABASE_URL', "YOUR_SUPABASE_URL") # Replace placeholder or use env var
# Use Service Role Key for backend! Find it in Supabase Settings -> API
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', "YOUR_SUPABASE_SERVICE_ROLE_KEY") # Replace placeholder or use env var

# Add your Storage Bucket Name to configuration. MUST be the exact name.
SUPABASE_STORAGE_BUCKET = os.environ.get('SUPABASE_STORAGE_BUCKET', "YOUR_STORAGE_BUCKET_NAME") # <--- ADD YOUR SUPABASE STORAGE BUCKET NAME HERE or use env var

if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
    print("Warning: OPENAI_API_KEY not set or using placeholder. OpenAI calls will fail.")
if not SUPABASE_URL or SUPABASE_URL == "YOUR_SUPABASE_URL" or \
   not SUPABASE_KEY or SUPABASE_KEY == "YOUR_SUPABASE_SERVICE_ROLE_KEY":
    print("Warning: Supabase URL or Key not set or using placeholders. Supabase operations may fail.")
    # Decide if this is a critical error that should prevent startup
    # exit(1)
if not SUPABASE_STORAGE_BUCKET or SUPABASE_STORAGE_BUCKET == "YOUR_STORAGE_BUCKET_NAME":
     print("Warning: SUPABASE_STORAGE_BUCKET name not set or using placeholder. Cannot process files from Storage.")
     # Decide if this is a critical error
     # exit(1)

openai.api_key = OPENAI_API_KEY
try:
    # Initialize the global supabase client instance using the Service Role Key
    # This client is used for both DB operations AND Storage downloads from backend
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Supabase client initialized in app.py.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize Supabase client in app.py: {e}")
    traceback.print_exc()
    supabase = None # Ensure it's None if initialization fails
    # Decide if this is a critical error
    # exit(1)

# Path to ChromaDB settings
# Use an environment variable for the Chroma DB path for deployment flexibility
CHROMA_DB_PATH = os.environ.get('CHROMA_DB_PATH', "./chroma_db_store") # Directory for persistent DB
CHROMA_COLLECTION_NAME = "tnpsc_brain_collection" # More descriptive collection name
EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'

# Global variables for initialized clients/models
embedding_model = None
chroma_client = None
chroma_collection = None
# Supabase client is also a global variable initialized above


# --- NLTK Setup ---
try:
    # Check if the 'punkt' tokenizer is already available locally
    nltk.data.find('tokenizers/punkt')
    print("NLTK 'punkt' tokenizer found.")
except LookupError:
    # If not found, try to download it
    print("NLTK 'punkt' tokenizer not found. Attempting to download...")
    try:
        # Download the tokenizer data quietly to the default NLTK data directory
        # This happens during Docker build or first run if not present
        nltk.download('punkt', quiet=True)
        print("'punkt' tokenizer downloaded successfully.")
    except Exception as download_error:
        # Catch any error during the download itself (e.g., network issues)
        print(f"Error during NLTK 'punkt' download: {download_error}")
        # If NLTK is essential for chunking, you might want to exit or raise the error
        # raise download_error # Re-raise the exception
        # exit(1) # Exit if download failure is critical

except Exception as other_error:
     # Catch any other unexpected errors when trying to find NLTK data
     print(f"An unexpected error occurred finding NLTK 'punkt': {other_error}")
     # If NLTK is essential, consider exiting
     # exit(1)


# --- Helper Functions (PDF Processing, Chunking) ---

# --- UPDATED extract_text_from_pdf function ---
def extract_text_from_pdf(pdf_source_path, book_id="Unknown Book"):
    """
    Extracts text from a PDF file, handling local file paths,
    Supabase Storage paths, or potentially public URLs.

    Args:
        pdf_source_path (str): The path to the PDF. Can be a local file path,
                                a Supabase Storage path (e.g., 'pdfs/book.pdf'),
                                or potentially a public URL.
        book_id (str): The ID of the book being processed (for logging).

    Returns:
        list: A list of tuples (page_num, text) for pages with extracted text,
              or None if extraction failed.
    """
    pages_data = []
    temp_file_path = None # Path for a temporary file if needed
    file_obj = None # Variable to hold the file object open for PyPDF2

    print(f"Attempting to extract text for book '{book_id}' from source: '{pdf_source_path}'")

    try:
        # --- Determine Source Type (Simple Heuristics) ---
        # This logic might need refinement based on your exact filepath data
        # A more robust way might be to store source_type in the DB table
        is_url = bool(urlparse(pdf_source_path).scheme) # Check if it has a URL scheme (http, https, etc.)
        is_storage_path = not is_url # Assume if not a URL, it's a storage path (for deployed) or local

        # On a deployed server, we expect paths to be storage paths or URLs.
        # On a local dev machine, they might still be local paths.
        # A simple check if the file exists locally first can differentiate,
        # or rely on the assumption that deployed filepaths are remote.
        # Let's try checking for local file existence first as a fallback/dev aid.
        is_local = os.path.exists(pdf_source_path)


        if is_local:
            print(f"Detected source as local file.")
            file_to_open_path = pdf_source_path
            file_obj = open(file_to_open_path, 'rb') # Open local file directly

        elif is_url:
            print(f"Detected source as URL.")
            # --- Download from URL ---
            # This requires the 'requests' library (added import above)
            try:
                print(f"Downloading from URL: {pdf_source_path}")
                response = requests.get(pdf_source_path, stream=True)
                response.raise_for_status() # Raise an exception for bad status codes (400s, 500s)
                
                # Check if the content type is likely a PDF (optional)
                content_type = response.headers.get('Content-Type', '').lower()
                if 'pdf' not in content_type and mimetypes.guess_type(pdf_source_path)[0] != 'application/pdf':
                     print(f"Warning: Downloaded content type may not be PDF: {content_type}")


                # Save content to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    # Use response.iter_content for potentially large files
                    for chunk in response.iter_content(chunk_size=8192):
                        temp_file.write(chunk)
                    temp_file_path = temp_file.name # Store the path of the temporary file

                file_to_open_path = temp_file_path
                print(f"Downloaded URL content and saved to temporary file: {file_to_open_path}")

                # Open the temporary file for PyPDF2 to read
                file_obj = open(file_to_open_path, 'rb')

            except Exception as url_e:
                 print(f"Error downloading PDF from URL '{pdf_source_path}' for book '{book_id}': {url_e}")
                 traceback.print_exc()
                 return None # Return None if download fails

        elif is_storage_path: # Assume it's a Supabase Storage path
            print(f"Detected source as potential Supabase Storage path.")
            if not supabase:
                 print(f"Error: Supabase client not available to download PDF '{pdf_source_path}' for book '{book_id}'.")
                 return None
            if not SUPABASE_STORAGE_BUCKET:
                 print(f"Error: SUPABASE_STORAGE_BUCKET is not configured. Cannot download '{pdf_source_path}'.")
                 return None

            # --- Use the global Supabase client (with Service Role Key) to download from Storage ---
            print(f"Downloading from Supabase Storage bucket '{SUPABASE_STORAGE_BUCKET}': '{pdf_source_path}'")
            try:
                # Supabase download returns bytes of the file content
                # Use the 'from_' method as 'from' is a Python keyword
                file_content = supabase.storage.from_(SUPABASE_STORAGE_BUCKET).download(pdf_source_path)

                if not file_content:
                    print(f"Error: Downloaded empty content for '{pdf_source_path}' from storage for book '{book_id}'.")
                    return None

                # Create a temporary file to write the downloaded content
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(file_content)
                    temp_file_path = temp_file.name # Store the path of the temporary file

                file_to_open_path = temp_file_path
                print(f"Downloaded Storage content and saved to temporary file: {file_to_open_path}")

                # Open the temporary file for PyPDF2 to read
                file_obj = open(file_to_open_path, 'rb')

            except Exception as download_e:
                 print(f"Error downloading or saving PDF from Supabase Storage '{pdf_source_path}' for book '{book_id}': {download_e}")
                 traceback.print_exc()
                 return None

        else:
            print(f"Error: Could not determine source type for '{pdf_source_path}' for book '{book_id}'.")
            return None


        # --- PyPDF2 Processing (Existing Logic) ---
        # Use file_obj which is open(..., 'rb') regardless of source type if successful
        if file_obj:
            try:
                reader = PyPDF2.PdfReader(file_obj)
                if not reader.pages:
                    print(f"Warning: No pages found in PDF source: '{pdf_source_path}' for book '{book_id}'.")
                    # No need to close file_obj here, finally block handles it
                    return None

                # Check if reader is encrypted and can be decrypted (optional, might need password)
                # if reader.is_encrypted:
                #    try:
                #        reader.decrypt('') # Try with empty password
                #        print(f"Decrypted PDF for book '{book_id}'.")
                #    except Exception as decrypt_e:
                #        print(f"Warning: PDF for book '{book_id}' is encrypted and could not be decrypted: {decrypt_e}")
                       # return None # Or process only non-encrypted pages if possible


                for page_num in range(len(reader.pages)):
                    try:
                        page = reader.pages[page_num]
                        # Add timeout to extraction in case of problematic pages (optional)
                        # text = page.extract_text(timeout=5) # PyPDF2 might not support timeout directly, requires threading/multiprocessing
                        text = page.extract_text()
                        if text and text.strip():
                            pages_data.append((page_num + 1, text))
                        else:
                            print(f"Warning: No text extracted from page {page_num + 1} of source: '{pdf_source_path}' for book '{book_id}'.")
                    except Exception as e:
                        print(f"Warning: Error extracting text from page {page_num + 1} of source '{pdf_source_path}' for book '{book_id}': {e}")
                        traceback.print_exc() # Print traceback for page errors


            except Exception as pypdf_e:
                 print(f"Error reading PDF with PyPDF2 from source '{pdf_source_path}' for book '{book_id}': {pypdf_e}")
                 traceback.print_exc()
                 return None # Return None if PyPDF2 fails to read the file

            finally:
                # Ensure the file object is closed after processing
                if file_obj and not file_obj.closed:
                     file_obj.close()
                     # print(f"Closed file object for source: {pdf_source_path}") # Uncomment for detailed logging


    except FileNotFoundError:
         # This specific error is mainly for source_type='local' if file_to_open path doesn't exist BEFORE open()
         print(f"Error: Local PDF file not found at {pdf_source_path} for book '{book_id}'.")
         return None
    except Exception as e:
        # Catch any other unexpected errors during the entire process
        print(f"An unexpected error occurred processing PDF source '{pdf_source_path}' for book '{book_id}': {e}")
        traceback.print_exc()
        return None
    finally:
        # --- Clean up the temporary file if one was created ---
        # This runs even if exceptions occurred above
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                # print(f"Cleaned up temporary file: {temp_file_path}") # Uncomment for detailed logging
            except Exception as cleanup_e:
                print(f"Error cleaning up temporary file {temp_file_path}: {cleanup_e}")
                traceback.print_exc()


    # Return the extracted data if any pages were processed, otherwise None
    return pages_data if pages_data else None


def chunk_by_sentence_window_with_source(pages_data, window_size=5, overlap=2):
    """
    Chunks text from page data into overlapping sentence windows using NLTK.
    """
    chunks_with_source = []
    if not pages_data:
        return chunks_with_source

    # Check if NLTK's sent_tokenize is available (depends on 'punkt' being downloaded)
    # Check dir(nltk.tokenize) is safer than direct attribute access if download failed
    if 'sent_tokenize' not in dir(nltk.tokenize) or not callable(nltk.tokenize.sent_tokenize):
         print("Error: NLTK 'punkt' tokenizer is not available or callable. Cannot chunk by sentence.")
         # Fallback: Could chunk by paragraph or character count as an alternative
         # For now, returning empty chunks if sentence tokenization isn't possible
         return chunks_with_source

    for page_num, page_text in pages_data:
        # Ensure page_text is not None or empty before tokenizing
        if not page_text or not page_text.strip():
             continue

        try:
            # Perform sentence tokenization
            sentences = sent_tokenize(page_text)
            if not sentences:
                 # print(f"Warning: No sentences tokenized from page {page_num}") # Can be chatty
                 continue

            # Process sentences into overlapping windows
            effective_window_size = min(window_size, len(sentences))
            if effective_window_size <= 0: # Handle case with very few sentences less than window size
                 continue

            # Step size for sliding window, minimum 1 to avoid infinite loop
            step_size = max(1, effective_window_size - overlap)

            for i in range(0, len(sentences) - effective_window_size + 1, step_size):
                chunk = " ".join(sentences[i:i + effective_window_size])
                chunks_with_source.append((chunk, page_num))

            # Handle the last few sentences if they don't form a full window but extend past the last start point
            # This is often handled by the range calculation itself if step_size is > 0
            # If len(sentences) is small, the range might not execute, handle that.
            if len(sentences) < effective_window_size and len(sentences) > 0:
                 chunk = " ".join(sentences)
                 chunks_with_source.append((chunk, page_num))
            # Note: Simple range(..., step_size) naturally handles the end overlap in many cases
            # For instance, range(0, 10, 3) goes 0, 3, 6, 9. The last window starts at 9.
            # If window=5, sentences[9:14] is valid even if sentences only has 10 items.
            # The standard range loop is usually sufficient.

        except Exception as sent_error:
             print(f"Error tokenizing or chunking sentences on page {page_num}: {sent_error}")
             traceback.print_exc()
             # Decide how to handle this - skip page? Log and continue?

    return chunks_with_source


# --- Embedding and Storage Function ---
def generate_and_store_embeddings(book_details, collection, model):
    """
    Generates embeddings for a book's content and stores them in ChromaDB.
    Handles determining the source type (local, storage, url) based on filepath format.
    """
    # The 'filepath' from the DB now contains the path in Supabase Storage, a local path, or a URL
    pdf_source_path = book_details.get('filepath')
    book_id = str(book_details.get('book_id')) # Ensure book_id is a string for consistent ID generation
    book_title = book_details.get('title', os.path.basename(pdf_source_path) if pdf_source_path else "Unknown Title")

    if not pdf_source_path:
        print(f"Error: No filepath provided for book_id '{book_id}'. Skipping.")
        return False

    # --- Determine Source Type for extract_text_from_pdf ---
    # Simple heuristics:
    # Check if it looks like a URL
    is_url = bool(urlparse(pdf_source_path).scheme)

    # Check if it exists as a local file (useful for dev or mixed environments)
    is_local = os.path.exists(pdf_source_path)

    # If it's neither a detectable URL nor an existing local file, assume it's a storage path
    # This assumption is crucial for the deployed environment
    source_type_for_extraction = "storage" # Default to storage for deployment
    if is_local:
        source_type_for_extraction = "local"
    elif is_url:
         source_type_for_extraction = "url"


    print(f"Processing book: '{book_title}' (ID: {book_id}) from source '{pdf_source_path}' (Detected type: {source_type_for_extraction}) for collection '{collection.name}'")

    # Call the updated extraction function, passing the detected source type
    pages_data = extract_text_from_pdf(pdf_source_path, book_id=book_id, source_type=source_type_for_extraction) # <--- Pass source_type

    if not pages_data:
        print(f"Failed to extract text from '{pdf_source_path}' for book_id '{book_id}'. No embeddings will be generated.")
        return False

    # Ensure NLTK is available for chunking
    if 'sent_tokenize' not in dir(nltk.tokenize) or not callable(nltk.tokenize.sent_tokenize):
        print("Critical Error: NLTK 'punkt' tokenizer is not available. Cannot chunk text.")
        return False # Cannot proceed if chunking is essential

    chunks_with_source = chunk_by_sentence_window_with_source(pages_data)
    if not chunks_with_source:
        print(f"No valid text chunks found for '{pdf_source_path}'. No embeddings will be generated for book_id '{book_id}'.")
        return False

    print(f"Generated {len(chunks_with_source)} chunks from '{book_title}'. Now creating embeddings...")
    texts_to_embed = [text for text, _ in chunks_with_source]
    try:
        # model.encode returns a numpy array
        embeddings = model.encode(texts_to_embed, show_progress_bar=True)
    except Exception as e:
        print(f"Error generating embeddings for book_id '{book_id}': {e}")
        traceback.print_exc()
        return False

    documents = [text for text, _ in chunks_with_source]
    # Ensure page numbers are strings for metadata consistency
    page_numbers_for_chunks = [str(page) for _, page in chunks_with_source]

    # Generate unique IDs for Chroma DB. Include book_id and page number for source tracking.
    # Add a chunk index to ensure uniqueness if multiple chunks per page.
    ids = [f"{book_id}_page_{page_numbers_for_chunks[i]}_chunk_{i}" for i in range(len(documents))]

    # Create metadata for each chunk
    metadatas = [{
        "page": page_numbers_for_chunks[i],
        "source_document_id": book_id,
        "book_title": book_title,
        # Store the original source path or just the filename for metadata
        "original_source_path": pdf_source_path, # Or os.path.basename(pdf_source_path)
        "original_filename": os.path.basename(pdf_source_path) # Store just filename
    } for i in range(len(documents))]

    # Add embeddings, documents, metadata, and IDs to Chroma DB in batches
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
            # ChromaDB add expects lists
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
            # Decide if a single batch failure is critical enough to stop the whole book processing
            # For now, we print the error but continue trying subsequent batches
            # If a critical error occurs (like Chroma connection lost), it might fail later anyway


    print(f"Completed adding chunks for '{book_title}' (Book ID: {book_id}).")
    return True # Return True even if some batches failed, if you want to allow partial processing


# --- RAG Core Functions ---
def find_relevant_chunks_chroma(query_text, collection, model, top_n=5, similarity_threshold=0.5, where_filter=None):
    """
    Finds relevant text chunks in ChromaDB for a given query using embedding similarity.
    """
    if not collection:
        print("Error: Chroma collection is not available for querying.")
        return []
    if not model:
         print("Error: Embedding model is not loaded for querying.")
         return []

    try:
        # Encode the query text into an embedding vector
        query_embedding = model.encode([query_text]).tolist()

        # Define parameters for the ChromaDB query
        query_params = {
            "query_embeddings": query_embedding,
            "n_results": top_n,
            "include": ["documents", "metadatas", "distances"] # Include chunk text, metadata, and distance score
        }
        # Add filter if provided (e.g., filtering by source_document_id)
        if where_filter:
            query_params["where"] = where_filter
            print(f"Chroma query with filter: {where_filter}")

        # Execute the query against the Chroma collection
        results = collection.query(**query_params)

    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        traceback.print_exc()
        return []

    relevant_chunks_info = []
    # Process the results if successful and items are found
    if results and results.get('documents') and results['documents'][0]:
        # Iterate through the results (which are often nested lists)
        for i in range(len(results['documents'][0])):
            distance = results['distances'][0][i]
            # Calculate similarity score. For cosine space, it's 1 - distance.
            # If your collection uses 'l2', the calculation differs.
            similarity_score = 1 - distance # Assuming 'cosine' space

            # Filter chunks based on the similarity threshold
            if similarity_score >= similarity_threshold:
                metadata = results['metadatas'][0][i]
                relevant_chunks_info.append({
                    'chunk_text': results['documents'][0][i],
                    'page_number': metadata.get('page', 'N/A'), # Get page from metadata
                    'source_document_id': metadata.get('source_document_id', 'N/A'), # Get book ID
                    'book_title': metadata.get('book_title', 'N/A'), # Get book title
                    'similarity': round(similarity_score, 4) # Add similarity score
                })
            # else:
            #     print(f"Skipping chunk below threshold: {similarity_score} < {similarity_threshold}") # Uncomment for debugging


    # Sort the results by similarity score in descending order
    return sorted(relevant_chunks_info, key=lambda x: x['similarity'], reverse=True)

def get_answer_from_openai(question, context_chunks):
    """
    Generates an AI answer based on the question and provided context chunks.
    """
    # Check if OpenAI API key is configured
    if not openai.api_key or openai.api_key == "YOUR_OPENAI_API_KEY":
        print("Error: OpenAI API key not configured. Cannot generate answer.")
        return "Error: AI service is not configured."

    # Format the context chunks into a single string
    context_str = "\n\n---\n\n".join([chunk['chunk_text'] for chunk in context_chunks]) if context_chunks else "No specific relevant context found in the documents."

    # Define the system's role and instructions
    system_instruction = (
        "You are a helpful AI assistant specializing in TNPSC exam preparation materials. "
        "Your task is to answer the user's question accurately based *only* on the provided document context. "
        "Do not use outside knowledge. If the answer cannot be found in the context, state clearly that the information is not available in the provided documents. "
        "Provide concise and direct answers."
    )

    # Create the user prompt with the context and the question
    user_prompt = f"Context from documents:\n\"\"\"\n{context_str}\n\"\"\"\n\nQuestion: {question}\n\nAnswer:"

    try:
        # Make the call to the OpenAI Chat Completions API
        response = openai.chat.comions.create(
            model="gpt-3.5-turbo", # You can change to gpt-4 or other models if available
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2, # Controls randomness (lower is more focused)
            max_tokens=500 # Limit the length of the generated answer
             # top_p, frequency_penalty, presence_penalty could also be tuned
        )

        # Extract the generated text from the response
        generated_text = response.choices[0].message.content.strip()
        return generated_text

    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        traceback.print_exc()
        # Provide a user-friendly error message
        return f"Error generating answer from AI: {e}"


# --- System Initialization ---
def initialize_system():
    """
    Initializes the embedding model, ChromaDB client/collection, and processes
    any books in the database that have not been processed yet.
    This function should be called ONCE when the application starts.
    """
    global embedding_model, chroma_client, chroma_collection, supabase

    print("--- System Initialization Start ---")

    # Ensure Supabase client was initialized successfully
    if supabase is None:
        print("CRITICAL ERROR: Supabase client failed to initialize earlier. Cannot proceed with system initialization.")
        return False

    # 1. Load Embedding Model
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    try:
        # SentenceTransformer model files are downloaded to a cache directory (often ~/.cache/torch/sentence_transformers)
        # or /app/models if you set SENTENCE_TRANSFORMERS_HOME in Dockerfile
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}")
        traceback.print_exc()
        return False # Critical failure

    # 2. Initialize ChromaDB Client
    print(f"Initializing ChromaDB client at path: {CHROMA_DB_PATH}...")
    try:
        # Use PersistentClient to load data from the specified path
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        print("ChromaDB client initialized.")
        # You might add a check here if the path seems invalid or permissions are wrong
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize ChromaDB client at '{CHROMA_DB_PATH}': {e}")
        traceback.print_exc()
        return False # Critical failure

    # 3. Get or Create Collection
    print(f"Getting or creating ChromaDB collection: {CHROMA_COLLECTION_NAME}...")
    try:
        # Get the collection if it exists, create it if it doesn't
        chroma_collection = chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            # Specify the embedding space. Must match the type used by your model. Cosine is common.
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Chroma collection '{CHROMA_COLLECTION_NAME}' is ready (Current item count: {chroma_collection.count()}).")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to setup ChromaDB collection '{CHROMA_COLLECTION_NAME}': {e}")
        traceback.print_exc()
        return False # Critical failure

    # --- Check for necessary configuration before processing books ---
    if not SUPABASE_STORAGE_BUCKET or SUPABASE_STORAGE_BUCKET == "YOUR_STORAGE_BUCKET_NAME":
         print("Warning: SUPABASE_STORAGE_BUCKET name is not configured. Skipping book processing from Storage.")
         # Decide if lack of storage bucket name is a critical error or just skips initial processing
         # If processing on startup is essential for function, return False here
         # return False # Example if storage processing is critical

    # 4. Fetch books from Supabase DB and process if needed
    print("Fetching book list from Supabase database...")
    try:
        # Select all necessary fields from the 'books' table
        response = supabase.table('books')\
                           .select('book_id, title, filepath, processed, subject, author, version, last_processed_at')\
                           .execute()

        # Check for Supabase API errors during select
        if hasattr(response, 'error') and response.error:
            print(f"Error fetching books from Supabase: {response.error.message}")
            traceback.print_exc()
            # Decide if failure to fetch book list is critical
            # return False # Example if fetching book list is critical

        elif response.data is not None: # Check if data is successfully returned (can be empty list)
            books_list = response.data
            print(f"Found {len(books_list)} books in Supabase.")

            processed_count = 0
            skipped_count = 0
            failed_processing_count = 0

            for book_details in books_list:
                book_id_str = str(book_details.get('book_id'))
                book_title = book_details.get('title', "Untitled Book")
                book_filepath = book_details.get('filepath')

                # Determine if processing is needed:
                # - If 'processed' flag is false or null
                # - You might add other conditions: e.g., if file modified date > last_processed_at, if version changed, etc.
                needs_processing = not book_details.get('processed', False) # Default to False if 'processed' is missing

                # Also skip if filepath is missing, as we can't process it
                if not book_filepath:
                     print(f"Skipping book '{book_title}' (ID: {book_id_str}): No filepath provided in DB.")
                     skipped_count += 1
                     continue # Move to the next book

                if needs_processing:
                    print(f"\n--- Queuing Book for Processing: '{book_title}' (ID: {book_id_str}) ---")
                    print(f"  Source filepath: '{book_filepath}'")

                    # Call the function to generate and store embeddings
                    # This function now handles reading from different sources (local/storage/url)
                    if generate_and_store_embeddings(book_details, chroma_collection, embedding_model):
                        # If processing was successful, update the status in Supabase
                        print(f"Successfully processed and stored embeddings for: '{book_title}'")
                        try:
                            # Update 'processed' status and timestamp in Supabase DB
                            supabase.table('books').update({
                                'processed': True,
                                'last_processed_at': datetime.datetime.utcnow().isoformat()
                            }).eq('book_id', book_id_str).execute() # Use book_id_str (string) for the query filter
                            print(f"Updated 'processed' status for book '{book_id_str}' in Supabase.")
                            processed_count += 1
                        except Exception as e_supa:
                            print(f"Error updating Supabase 'processed' status for book '{book_id_str}': {e_supa}")
                            traceback.print_exc()
                            # Decide how to handle DB update failure - retry? Log?

                    else:
                        # If generate_and_store_embeddings returned False (indicating failure)
                        print(f"Failed to process book: '{book_title}' (ID: {book_id_str}). Embeddings were NOT generated or stored.")
                        failed_processing_count += 1
                        # You might want to log this failure or update the DB status differently (e.g., 'processing_failed' flag)
                else:
                    print(f"Book '{book_title}' (ID: {book_id_str}) already marked as processed. Skipping initial load.")
                    skipped_count += 1 # Count as skipped for this pass

            print(f"\n--- Initial Book Processing Summary ---")
            print(f"Total books in DB: {len(books_list)}")
            print(f"Processed during this run: {processed_count}")
            print(f"Skipped (already processed or no filepath): {skipped_count}")
            print(f"Failed processing: {failed_processing_count}")


        else:
            # Case where response.data is None and no explicit error
            print("Supabase response for books table did not contain data.")


    except Exception as e:
        # Catch unexpected errors during the Supabase fetch loop
        print(f"Error during Supabase book fetching or processing phase: {e}")
        traceback.print_exc()
        # Decide if failure to fetch/process books is critical
        # return False # Example if book processing is critical

    print("--- System Initialization End ---")
    # Return True if core components (embedding model, chroma, supabase) are ready,
    # even if book processing had warnings or failures. Adjust if book processing is mandatory for app function.
    return embedding_model is not None and chroma_client is not None and chroma_collection is not None and supabase is not None


# --- Removed Flask App Setup and Routes ---
# The Flask app instance, Swagger setup, and route definitions
# are now expected to be in your fi.py file.
# Remove the app = Flask(__name__) line
# Remove the @app.route(...) decorators and the functions that follow them
# Remove the swagger_spec function

# --- Removed Main Execution Block from app.py ---
# The __main__ block that runs the Flask app is now in fi.py.
# Remove the if __name__ == '__main__': block from here.