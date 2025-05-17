# --- START OF FILE fi.py (Updated for Supabase pgvector) ---
from dotenv import load_dotenv
import os # Add os import here for testing

print("Attempting to load .env file...")
loaded_successfully = load_dotenv()
print(f".env file loaded: {loaded_successfully}") # Should print True if .env was found and read

# Test if variables are loaded immediately
print(f"SUPABASE_URL from os.environ in fi.py (after load_dotenv): {os.environ.get('SUPABASE_URL')}")
print(f"SUPABASE_KEY from os.environ in fi.py (after load_dotenv): {os.environ.get('SUPABASE_KEY')[:5] if os.environ.get('SUPABASE_KEY') else None}...")

import datetime
import os
import uuid
import traceback
from flask import Flask, request, jsonify
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS

import app # Import the entire app module

app_fi = Flask(__name__)
CORS(app_fi)

SWAGGER_URL_FI = '/api/docs'
API_URL_FI = '/swagger.json'
swaggerui_blueprint_fi = get_swaggerui_blueprint(
    SWAGGER_URL_FI, API_URL_FI,
    config={'app_name': "TNPSC Brain Q&A API (Supabase pgvector)"}
)
app_fi.register_blueprint(swaggerui_blueprint_fi, url_prefix=SWAGGER_URL_FI)

def get_user_from_request():
    if app.supabase is None:
        print("Error: Supabase client not initialized in app.py.")
        return None, "Backend Supabase client not configured."

    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return None, "Authorization header is missing."

    try:
        scheme, token = auth_header.split()
        if scheme.lower() != 'bearer':
            return None, "Authorization scheme must be Bearer."
    except ValueError:
        return None, "Invalid Authorization header format (e.g., 'Bearer <token>')."

    try:
        user_response = app.supabase.auth.get_user(token)
        if user_response and user_response.user:
            print(f"Authenticated user: {user_response.user.id}")
            return user_response.user, None
        else:
            print("Supabase JWT verification returned no user or an unexpected response.")
            return None, "Token validation failed or token expired."
    except Exception as e:
        print(f"Error during JWT verification process: {e}")
        traceback.print_exc()
        if hasattr(e, 'message'): # GJSONDecodeError, AuthApiError might have message
            return None, f"Token validation error: {getattr(e, 'message', str(e))}"
        return None, f"Internal token validation error: {str(e)}"


@app_fi.route('/swagger.json')
def swagger_spec_fi():
    server_url = request.host_url.rstrip('/')
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "TNPSC Brain Q&A API (Supabase pgvector)",
            "version": "1.4.0", # Updated version
            "description": "Ask questions about TNPSC content, with user authentication and history, using Supabase pgvector."
        },
        "servers": [{"url": server_url}],
        "paths": {
            "/ask": {
                "post": {
                    "summary": "Ask a question (Authentication Required).",
                    "security": [{"bearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "question": {"type": "string", "example": "Types of soil?"},
                                        "top_n_chunks": {"type": "integer", "default": 5},
                                        "similarity_threshold": {"type": "number", "default": 0.5},
                                        "filter_book_ids": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "example": ["book1_id"],
                                            "description": "Optional list of book_ids (strings) to filter."
                                        }
                                    },
                                    "required": ["question"]
                                }
                            }
                        }
                    },
                     "responses": {
                        "200": {"description": "Answer with sources."},
                        "400": {"description": "Bad Request"},
                        "401": {"description": "Unauthorized"},
                        "500": {"description": "Internal Server Error"},
                        "503": {"description": "Service Unavailable (System not initialized)"}
                    }
                }
            },
            "/save_interaction": {
                "post": {
                    "summary": "Mark an interaction as saved/unsaved (Authentication Required).",
                    "security": [{"bearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "interaction_id": {"type": "string", "example": "uuid_here"},
                                        "is_saved": {"type": "boolean", "example": True}
                                    },
                                    "required": ["interaction_id", "is_saved"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {"description": "Save status updated."},
                        "400": {"description": "Bad Request"},
                         "401": {"description": "Unauthorized"},
                        "404": {"description": "Interaction not found."},
                        "500": {"description": "Internal Server Error"},
                        "503": {"description": "Supabase not initialized"}
                    }
                }
            },
            "/saved_interactions": {
                "get": {
                    "summary": "Retrieve saved interactions (Authentication Required).",
                    "security": [{"bearerAuth": []}],
                    "responses": {
                        "200": {"description": "List of saved interactions."},
                        "401": {"description": "Unauthorized"},
                        "500": {"description": "Internal Server Error"},
                        "503": {"description": "Service Unavailable (System not initialized)"}
                    }
                }
            },
             "/books": {
                 "get": {
                     "summary": "Retrieve list of study materials (Authentication Required).",
                     "security": [{"bearerAuth": []}],
                     "responses": {
                         "200": {"description": "List of books."},
                         "401": {"description": "Unauthorized"},
                         "500": {"description": "Internal Server Error"},
                         "503": {"description": "Supabase not initialized"}
                     }
                 }
             },
            "/reprocess_book/{book_id_to_reprocess}": {
                "post": {
                    "summary": "Force reprocessing of a specific book (Authentication Required).",
                    "security": [{"bearerAuth": []}],
                    "parameters": [
                        {
                            "name": "book_id_to_reprocess", "in": "path", "required": True,
                            "description": "The book_id (string) of the book to reprocess.",
                            "schema": {"type": "string"}
                        }
                    ],
                    "responses": {
                        "200": {"description": "Book reprocessing initiated."},
                        "401": {"description": "Unauthorized"},
                        "404": {"description": "Book not found."},
                        "500": {"description": "Error during reprocessing."},
                        "503": {"description": "Service Unavailable"}
                    }
                }
            }
        },
        "components": {
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT"
                }
            }
        }
    }
    return jsonify(spec)


@app_fi.route('/books', methods=['GET'])
def get_books_route():
    user, auth_error = get_user_from_request()
    if auth_error:
        return jsonify({"error": auth_error}), 401

    if app.supabase is None:
         return jsonify({"error": "Backend Supabase client not configured."}), 503
    print(f"User {user.id} requesting book list.")
    try:
        response = app.supabase.table('books')\
                     .select('book_id, title, filepath, processed, subject, author, version, last_processed_at')\
                     .order('title', ascending=True)\
                     .execute()
        if hasattr(response, 'error') and response.error:
             print(f"Supabase select books error: {response.error.message if response.error else 'Unknown error'}")
             return jsonify({"error": response.error.message if response.error else "DB error"}), 500
        return jsonify(response.data), 200
    except Exception as e:
        print(f"Error fetching books: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app_fi.route('/reprocess_book/<book_id_to_reprocess>', methods=['POST'])
def reprocess_single_book_route(book_id_to_reprocess):
    user, auth_error = get_user_from_request()
    if auth_error:
        return jsonify({"error": auth_error}), 401

    # Check for Supabase client and embedding model (Chroma collection check removed)
    if app.supabase is None or app.embedding_model is None:
        return jsonify({"error": "System components (Supabase client or embedding model) not fully initialized."}), 503

    print(f"User {user.id} attempting to reprocess book_id: {book_id_to_reprocess}")
    try:
        # Ensure book_id_to_reprocess is treated as a string for DB queries
        book_id_str = str(book_id_to_reprocess)
        response = app.supabase.table('books').select('book_id, title, filepath').eq('book_id', book_id_str).maybe_single().execute()

        if response.data:
            book_details = response.data
            book_title = book_details.get('title', book_id_str)
            print(f"Found book: {book_title}")

            try:
                print(f"Deleting existing embeddings for book '{book_title}' (ID: {book_id_str}) from Supabase table '{app.SUPABASE_EMBEDDINGS_TABLE_NAME}'...")
                delete_response = app.supabase.table(app.SUPABASE_EMBEDDINGS_TABLE_NAME)\
                    .delete()\
                    .eq('book_id', book_id_str)\
                    .execute()
                
                if hasattr(delete_response, 'error') and delete_response.error:
                    print(f"Warning: Error deleting embeddings for book '{book_title}': {delete_response.error.message if delete_response.error else 'Unknown DB error'}")
                    # Depending on policy, you might stop here or continue
                else:
                    # delete_response.data might contain the deleted records or count, depending on Supabase version and settings
                    deleted_count = len(delete_response.data) if delete_response.data else "an unknown number of"
                    print(f"Deleted {deleted_count} existing embeddings for book '{book_title}'.")

            except Exception as e_delete:
                print(f"Exception during deletion of embeddings for book '{book_title}': {e_delete}")
                traceback.print_exc()
                # Continue reprocessing even if deletion had issues, as new embeddings might overwrite or co-exist if not properly deleted

            # Pass model directly, no collection object needed
            if app.generate_and_store_embeddings(book_details, app.embedding_model):
                print(f"Successfully reprocessed book: '{book_title}'")
                try:
                    update_book_status_resp = app.supabase.table('books').update({
                        'processed': True,
                        'last_processed_at': datetime.datetime.utcnow().isoformat()
                    }).eq('book_id', book_id_str).execute()

                    if hasattr(update_book_status_resp, 'error') and update_book_status_resp.error:
                         print(f"Error updating 'books' table status after reprocessing book '{book_id_str}': {update_book_status_resp.error.message if update_book_status_resp.error else 'Unknown DB error'}")
                         # Even if this fails, the core reprocessing succeeded.
                    
                    return jsonify({"message": f"Book '{book_title}' reprocessed successfully."}), 200
                except Exception as e_supa_update:
                    print(f"Error updating Supabase 'books' table status after reprocessing: {e_supa_update}")
                    return jsonify({"error": f"Failed to update DB status for '{book_title}', but reprocessing might be complete."}), 500 # Or 200 with a warning
            else:
                print(f"Failed to reprocess book: '{book_title}'.")
                # Potentially set 'processed' to False again if it's critical
                # app.supabase.table('books').update({'processed': False}).eq('book_id', book_id_str).execute()
                return jsonify({"error": f"Failed to reprocess book '{book_title}'."}), 500
        else:
            if hasattr(response, 'error') and response.error:
                print(f"Error fetching book {book_id_str}: {response.error.message if response.error else 'Unknown DB error'}")
                return jsonify({"error": f"DB error fetching book: {response.error.message if response.error else 'Unknown DB error'}"}), 500
            print(f"Book with ID '{book_id_str}' not found.")
            return jsonify({"error": f"Book with ID '{book_id_str}' not found."}), 404

    except Exception as e:
        print(f"Error reprocessing book {book_id_to_reprocess}: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app_fi.route('/ask', methods=['POST'])
def ask_question_route():
    user, auth_error = get_user_from_request()
    if auth_error:
        return jsonify({"error": auth_error}), 401
    user_id = user.id
    print(f"User {user_id} asking question.")

    # Check for Supabase client and embedding model (Chroma collection check removed)
    if app.supabase is None or app.embedding_model is None:
        return jsonify({"error": "System (Supabase client or embedding model) not initialized."}), 503

    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question'"}), 400

    question = data['question']
    top_n = data.get('top_n_chunks', 5)
    similarity_threshold = data.get('similarity_threshold', 0.5)
    
    # filter_book_ids should be a list of strings or None
    filter_book_ids_input = data.get('filter_book_ids', None)
    valid_filter_book_ids = None
    if filter_book_ids_input and isinstance(filter_book_ids_input, list):
        # Ensure all elements are strings, filter out empty or non-string elements if any
        valid_filter_book_ids = [str(bid) for bid in filter_book_ids_input if isinstance(bid, (str, int, float)) and str(bid).strip()]
        if not valid_filter_book_ids: # If list becomes empty after filtering
            valid_filter_book_ids = None
    
    # Call the Supabase version of find_relevant_chunks
    relevant_chunks = app.find_relevant_chunks_supabase(
        question, app.embedding_model, top_n, similarity_threshold, valid_filter_book_ids
    )
    ai_answer = app.get_answer_from_openai(question, relevant_chunks)
    interaction_id = str(uuid.uuid4())
    try:
        chunks_metadata_for_db = []
        if relevant_chunks: # Ensure relevant_chunks is not None or empty
            chunks_metadata_for_db = [
                {
                    'page_number': c.get('page_number'), 
                    'source_document_id': c.get('source_document_id'), 
                    'book_title': c.get('book_title'),
                    'similarity': c.get('similarity') # Storing similarity might be useful
                } for c in relevant_chunks
            ]
            # Filter out entries where source_document_id is missing (important for DB consistency)
            chunks_metadata_for_db = [m for m in chunks_metadata_for_db if m.get('source_document_id') not in [None, '', 'N/A']]

        insert_data = {
            'id': interaction_id, 'user_id': str(user_id), 'question_text': question,
            'answer_text': ai_answer, 'relevant_chunks_metadata': chunks_metadata_for_db, # Use the filtered list
            'similarity_threshold_used': similarity_threshold, 
            'filter_book_ids_used': valid_filter_book_ids # Store the actual filter used
        }
        if app.supabase:
            interaction_response = app.supabase.table('interactions').insert(insert_data).execute()
            if hasattr(interaction_response, 'error') and interaction_response.error:
                 print(f"Supabase insert interaction error: {interaction_response.error.message if interaction_response.error else 'Unknown DB error'}")
        else:
            print("Supabase client not available for interaction logging.")
    except Exception as e:
        print(f"Error storing interaction: {e}")
        traceback.print_exc()

    return jsonify({
        "interaction_id": interaction_id, "question": question,
        "answer": ai_answer, "relevant_chunks": relevant_chunks
    })

@app_fi.route('/save_interaction', methods=['POST'])
def save_interaction_route():
    user, auth_error = get_user_from_request()
    if auth_error:
        return jsonify({"error": auth_error}), 401
    user_id = user.id
    print(f"User {user_id} saving interaction.")

    if app.supabase is None:
        return jsonify({"error": "Backend Supabase client not configured."}), 503

    data = request.get_json()
    if not data or not all(k in data for k in ['interaction_id', 'is_saved']):
        return jsonify({"error": "Missing data (interaction_id, is_saved)."}), 400

    interaction_id = data['interaction_id']
    is_saved = bool(data['is_saved'])
    try:
        result = app.supabase.table('interactions').update({'is_saved': is_saved})\
            .eq('id', str(interaction_id)).eq('user_id', str(user_id)).execute() # Ensure IDs are strings

        if hasattr(result, 'error') and result.error:
            print(f"Supabase update error: {result.error.message if result.error else 'Unknown DB error'}")
            # PGRST116 is "0 rows in result" which means not found or no update needed.
            if 'PGRST116' in str(result.error.message if result.error else ""):
                 return jsonify({"error": "Interaction not found for this user or no change required."}), 404
            return jsonify({"error": result.error.message if result.error else "DB error"}), 500

        # supabase-py v2 update response.data contains the updated records
        if result.data and len(result.data) > 0:
             return jsonify({"message": "Save status updated."}), 200
        else: # This case implies the record was not found for the user_id and interaction_id
             return jsonify({"error": "Interaction not found for this user or no update was performed."}), 404
    except Exception as e:
        print(f"Error saving interaction: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app_fi.route('/saved_interactions', methods=['GET'])
def get_saved_interactions_route():
    user, auth_error = get_user_from_request()
    if auth_error:
        return jsonify({"error": auth_error}), 401
    user_id = user.id
    print(f"User {user_id} requesting saved interactions.")

    if app.supabase is None:
        return jsonify({"error": "Backend Supabase client not configured."}), 503
    try:
        result = app.supabase.table('interactions')\
            .select("*").eq('user_id', str(user_id)).eq('is_saved', True)\
            .order('created_at', desc=True).execute() # Ensure user_id is string

        if hasattr(result, 'error') and result.error:
            print(f"Supabase select error: {result.error.message if result.error else 'Unknown DB error'}")
            return jsonify({"error": result.error.message if result.error else "DB error"}), 500
        return jsonify(result.data), 200
    except Exception as e:
        print(f"Error fetching saved interactions: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app_fi.route('/')
def index():
    return jsonify({"message": "TNPSC Q&A API (Supabase pgvector) running. See /api/docs for Swagger UI."})

if __name__ == '__main__':
    print("--- Starting fi.py system initialization ---")
    if app.initialize_system():
        # Check for Supabase client and embedding model (Chroma collection check removed)
        if app.supabase and app.embedding_model:
            print("--- fi.py system initialization complete. Starting Flask server ---")
            port = int(os.environ.get('PORT', 5001))
            print(f"Flask application (dev server) attempting to run on http://0.0.0.0:{port}")
            print(f"Swagger API docs available at http://0.0.0.0:{port}{SWAGGER_URL_FI}")
            app_fi.run(debug=True, port=port, host='0.0.0.0')
        else:
            print("--- fi.py system initialization failed (Supabase client or embedding model missing). Flask server will not start ---")
    else:
        print("--- Critical failure during app.initialize_system. Flask server will not start ---")

# --- END OF FILE fi.py (Updated for Supabase pgvector) ---