# --- START OF FILE fi.py (Updated) ---

import datetime
import os
import uuid
import traceback
from flask import Flask, request, jsonify
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS

import app # Import the entire app module

app_fi = Flask(__name__)
CORS(app_fi) # Allow all origins for now, refine for production

SWAGGER_URL_FI = '/api/docs'
API_URL_FI = '/swagger.json'
swaggerui_blueprint_fi = get_swaggerui_blueprint(
    SWAGGER_URL_FI, API_URL_FI,
    config={'app_name': "TNPSC Brain Q&A API (Authenticated & Deployed)"}
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
        # ******** CORRECTED SUPABASE AUTH CALL ********
        # For supabase-py v2.x, use supabase.auth.get_user(token)
        user_response = app.supabase.auth.get_user(token)
        # **********************************************

        if user_response and user_response.user:
            print(f"Authenticated user: {user_response.user.id}")
            return user_response.user, None
        # supabase-py v2 get_user() doesn't have an .error attribute on the response directly for JWT errors
        # It raises an exception (e.g., gotrue.errors.AuthApiError) or returns a response with user=None if token is invalid
        else:
            # This case might be hit if token is invalid but doesn't raise an exception
            print("Supabase JWT verification returned no user or an unexpected response.")
            return None, "Token validation failed or token expired."

    except Exception as e: # Catching generic Exception, might need to be more specific for gotrue errors if you want to parse them
        print(f"Error during JWT verification process: {e}")
        traceback.print_exc()
        # Check if it's a Supabase specific auth error
        if hasattr(e, 'message'):
            return None, f"Token validation error: {e.message}"
        return None, f"Internal token validation error: {str(e)}"


@app_fi.route('/swagger.json')
def swagger_spec_fi():
    server_url = request.host_url.rstrip('/')
    # Your Swagger spec seems mostly fine, just ensure paths and components are correct.
    # (Keeping your existing spec for brevity, ensure it matches all endpoints and auth.)
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "TNPSC Brain Q&A API (Authenticated)",
            "version": "1.3.1",
            "description": "Ask questions about TNPSC content, with user authentication and history."
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
                                            "description": "Optional list of book_ids to filter."
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
                            "description": "The book_id of the book to reprocess.",
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

    if app.supabase is None or app.chroma_collection is None or app.embedding_model is None:
        return jsonify({"error": "System components not fully initialized."}), 503

    print(f"User {user.id} attempting to reprocess book_id: {book_id_to_reprocess}")
    try:
        response = app.supabase.table('books').select('book_id, title, filepath').eq('book_id', str(book_id_to_reprocess)).maybe_single().execute() # Use maybe_single for safety

        if response.data:
            book_details = response.data
            book_title = book_details.get('title', book_id_to_reprocess)
            print(f"Found book: {book_title}")

            try:
                print(f"Deleting existing chunks for book '{book_title}' (ID: {book_id_to_reprocess})...")
                app.chroma_collection.delete(where={"source_document_id": str(book_id_to_reprocess)})
                print(f"Deleted existing chunks for book '{book_title}'.")
            except Exception as e_delete:
                print(f"Warning: Could not delete all chunks for book '{book_title}': {e_delete}")
                # Continue reprocessing even if deletion had issues

            if app.generate_and_store_embeddings(book_details, app.chroma_collection, app.embedding_model):
                print(f"Successfully reprocessed book: '{book_title}'")
                try:
                    app.supabase.table('books').update({
                        'processed': True,
                        'last_processed_at': datetime.datetime.utcnow().isoformat()
                    }).eq('book_id', str(book_id_to_reprocess)).execute()
                    return jsonify({"message": f"Book '{book_title}' reprocessed successfully."}), 200
                except Exception as e_supa:
                    print(f"Error updating Supabase status after reprocessing: {e_supa}")
                    return jsonify({"error": f"Failed to update DB status for '{book_title}'."}), 500
            else:
                print(f"Failed to reprocess book: '{book_title}'.")
                return jsonify({"error": f"Failed to reprocess book '{book_title}'."}), 500
        else:
            # Handle cases where book is not found or error in fetching
            if hasattr(response, 'error') and response.error:
                print(f"Error fetching book {book_id_to_reprocess}: {response.error.message}")
                return jsonify({"error": f"DB error fetching book: {response.error.message}"}), 500
            print(f"Book with ID '{book_id_to_reprocess}' not found.")
            return jsonify({"error": f"Book with ID '{book_id_to_reprocess}' not found."}), 404

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

    if app.chroma_collection is None or app.embedding_model is None:
        return jsonify({"error": "System not initialized."}), 503

    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question'"}), 400

    question = data['question']
    top_n = data.get('top_n_chunks', 5)
    similarity_threshold = data.get('similarity_threshold', 0.5)
    filter_book_ids = data.get('filter_book_ids', None)
    where_filter = None
    if filter_book_ids and isinstance(filter_book_ids, list) and len(filter_book_ids) > 0:
        str_filter_book_ids = [str(bid) for bid in filter_book_ids]
        if len(str_filter_book_ids) == 1:
            where_filter = {"source_document_id": str_filter_book_ids[0]}
        else:
            where_filter = {"source_document_id": {"$in": str_filter_book_ids}}

    relevant_chunks = app.find_relevant_chunks_chroma(
        question, app.chroma_collection, app.embedding_model, top_n, similarity_threshold, where_filter
    )
    ai_answer = app.get_answer_from_openai(question, relevant_chunks)
    interaction_id = str(uuid.uuid4())
    try:
        chunks_metadata = [{'page_number': c.get('page_number'), 'source_document_id': c.get('source_document_id'), 'book_title': c.get('book_title')} for c in relevant_chunks]
        chunks_metadata = [m for m in chunks_metadata if m.get('source_document_id') not in [None, '', 'N/A']]

        insert_data = {
            'id': interaction_id, 'user_id': user_id, 'question_text': question,
            'answer_text': ai_answer, 'relevant_chunks_metadata': chunks_metadata,
            'similarity_threshold_used': similarity_threshold, 'filter_book_ids_used': filter_book_ids
        }
        if app.supabase:
            response = app.supabase.table('interactions').insert(insert_data).execute()
            if hasattr(response, 'error') and response.error:
                 print(f"Supabase insert error: {response.error.message if response.error else 'Unknown error'}")
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
            .eq('id', interaction_id).eq('user_id', user_id).execute()

        if hasattr(result, 'error') and result.error:
            print(f"Supabase update error: {result.error.message if result.error else 'Unknown error'}")
            if 'PGRST116' in str(result.error): # Simple check for "Not Found" like error code
                 return jsonify({"error": "Interaction not found for this user."}), 404
            return jsonify({"error": result.error.message if result.error else "DB error"}), 500

        if result.data and len(result.data) > 0:
             return jsonify({"message": "Save status updated."}), 200
        else:
             return jsonify({"error": "Interaction not found for this user or no change needed."}), 404
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
            .select("*").eq('user_id', user_id).eq('is_saved', True)\
            .order('created_at', desc=True).execute()
        if hasattr(result, 'error') and result.error:
            print(f"Supabase select error: {result.error.message if result.error else 'Unknown error'}")
            return jsonify({"error": result.error.message if result.error else "DB error"}), 500
        return jsonify(result.data), 200
    except Exception as e:
        print(f"Error fetching saved interactions: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app_fi.route('/')
def index():
    return jsonify({"message": "TNPSC Q&A API (Authenticated) running. See /api/docs for Swagger UI."})

if __name__ == '__main__':
    print("--- Starting fi.py system initialization ---")
    if app.initialize_system():
        if app.chroma_collection and app.embedding_model and app.supabase:
            print("--- fi.py system initialization complete. Starting Flask server ---")
            # ******** MODIFIED FOR RENDER ********
            port = int(os.environ.get('PORT', 5001)) # Use PORT from env, default to 5001 for local
            # For production, Gunicorn will handle this. This `app_fi.run` is for local dev.
            # When deploying to Render, the Gunicorn command will be `gunicorn -w 4 -b 0.0.0.0:$PORT fi:app_fi`
            print(f"Flask application (dev server) attempting to run on http://0.0.0.0:{port}")
            print(f"Swagger API docs available at http://0.0.0.0:{port}{SWAGGER_URL_FI}")
            app_fi.run(debug=True, port=port, host='0.0.0.0')
            # ***********************************
        else:
            print("--- fi.py system initialization failed (core components missing). Flask server will not start ---")
    else:
        print("--- Critical failure during app.initialize_system. Flask server will not start ---")

# --- END OF FILE fi.py (Updated) ---