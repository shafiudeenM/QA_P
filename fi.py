# --- START OF FILE fi.py (Updated) ---

import datetime
import os
import uuid
import traceback
from flask import Flask, request, jsonify
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS # Make sure Flask-CORS is imported

# --- Import the entire app module ---
# This brings in configuration, initialization, and core functions from app.py
import app

# --- Flask Setup ---
# Create the single Flask application instance for your API
app_fi = Flask(__name__)

# Add CORS to allow requests from your frontend origin
# For local development, CORS(app_fi) or CORS(app_fi, resources={r"/*": {"origins": "*"}}) is fine.
# For production, configure origins securely: CORS(app_fi, origins=["https://your-frontend-domain.com"])
CORS(app_fi)

# --- Swagger UI Setup ---
SWAGGER_URL_FI = '/api/docs'
API_URL_FI = '/swagger.json'
swaggerui_blueprint_fi = get_swaggerui_blueprint(
    SWAGGER_URL_FI, API_URL_FI,
    config={'app_name': "TNPSC Brain Q&A API (Authenticated & Deployed)"}
)
app_fi.register_blueprint(swaggerui_blueprint_fi, url_prefix=SWAGGER_URL_FI)

# --- Helper function to get user from token (Moved/Ensured in fi.py) ---
def get_user_from_request():
    """Extracts and validates Supabase user from Authorization: Bearer header."""
    # Use the supabase client initialized in app.py (app.supabase)
    if app.supabase is None:
        print("Error: Supabase client not initialized in app.py.")
        # Return an error that indicates a backend configuration issue, not user auth issue
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
        # Verify the token using the Supabase client initialized in app.py
        # This requires the service role key (app.supabase client)
        user_response = app.supabase.auth.get_user_from_jwt(token)

        if user_response and user_response.user:
            # Success: Token is valid and user object is available
            print(f"Authenticated user: {user_response.user.id}") # Log user ID for debugging
            return user_response.user, None
        elif user_response and user_response.error:
             # Supabase returned an error specific to the token (e.g., expired, invalid)
            print(f"Supabase JWT verification failed: {user_response.error.message}")
            return None, user_response.error.message # Pass the specific error message

        else:
             # Unexpected response structure or no user/error (should be rare)
             print("Supabase JWT verification returned unexpected response.")
             return None, "Token validation failed."

    except Exception as e:
        # Catch any other exceptions during the process (network error talking to Supabase auth)
        print(f"Error during JWT verification process: {e}")
        traceback.print_exc()
        return None, f"Internal token validation error: {str(e)}"


# --- Swagger JSON (Updated for Authentication) ---
@app_fi.route('/swagger.json')
def swagger_spec_fi():
    server_url = request.host_url.rstrip('/')
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "TNPSC Brain Q&A API (Authenticated)", # Updated title
            "version": "1.3.1", # New version
            "description": "Ask questions about TNPSC content, with user authentication and history."
        },
        "servers": [{"url": server_url}],
        "paths": {
            "/ask": {
                "post": {
                    "summary": "Ask a question based on pre-processed TNPSC documents (Authentication Required).", # Updated summary
                    "security": [{"bearerAuth": []}], # <-- Requires bearer auth
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "question": {"type": "string", "example": "What are the types of soil in Tamil Nadu?"},
                                        # REMOVED "anonymous_user_id"
                                        "top_n_chunks": {"type": "integer", "default": 5, "example": 5},
                                        "similarity_threshold": {"type": "number", "default": 0.5, "example": 0.6},
                                        "filter_book_ids": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "example": ["history_vol1", "geography_tn"],
                                            "description": "Optional list of book_ids (as strings) to restrict search."
                                        }
                                    },
                                    "required": ["question"] # "question" is required
                                }
                            }
                        }
                    },
                     "responses": {
                        "200": {
                             "description": "Answer with sources.",
                             "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "interaction_id": {"type": "string", "description": "Unique ID for this interaction."},
                                            "question": {"type": "string"},
                                            "answer": {"type": "string"},
                                            "relevant_chunks": {
                                                "type": "array",
                                                "items": {
                                                     "type": "object",
                                                     "properties": {
                                                         "chunk_text": {"type": "string"},
                                                         "page_number": {"type": "string"},
                                                         "source_document_id": {"type": "string"},
                                                         "book_title": {"type": "string"},
                                                         "similarity": {"type": "number"}
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                         },
                        "400": {"description": "Bad Request (e.g., missing question)"},
                        "401": {"description": "Unauthorized (missing or invalid token)"}, # <-- Added 401
                        "500": {"description": "Internal Server Error"},
                        "503": {"description": "Service Unavailable (System not initialized)"}
                    }
                }
            },
            "/save_interaction": {
                "post": {
                    "summary": "Mark an interaction as saved or unsaved (Authentication Required).", # Updated summary
                    "security": [{"bearerAuth": []}], # <-- Requires bearer auth
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        # REMOVED "anonymous_user_id"
                                        "interaction_id": {"type": "string", "example": "UUID_FROM_ASK_RESPONSE", "description": "The ID received in the /ask response."},
                                        "is_saved": {"type": "boolean", "example": True, "description": "Set to true to save, false to unsave."}
                                    },
                                    "required": ["interaction_id", "is_saved"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {"description": "Save status updated."},
                        "400": {"description": "Bad Request"},
                         "401": {"description": "Unauthorized (missing or invalid token)"}, # <-- Added 401
                        "404": {"description": "Interaction not found (or user/interaction ID mismatch)."},
                        "500": {"description": "Internal Server Error"},
                        "503": {"description": "Supabase not initialized"}
                    }
                }
            },
            "/saved_interactions": {
                "get": {
                    "summary": "Retrieve saved interactions for the authenticated user (Authentication Required).", # Updated summary
                    "security": [{"bearerAuth": []}], # <-- Requires bearer auth
                    "parameters": [
                        # REMOVED "anonymous_user_id" query parameter
                         # Parameter list is now empty or can be used for other filters
                    ],
                    "responses": {
                        "200": {
                            "description": "List of saved interactions.",
                             "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {
                                             "type": "object", # Define the structure of an interaction item
                                             "properties": {
                                                "id": {"type": "string"},
                                                "created_at": {"type": "string", "format": "date-time"},
                                                "user_id": {"type": "string"}, # Should be user_id now
                                                "question_text": {"type": "string"},
                                                "answer_text": {"type": "string"},
                                                "relevant_chunks_metadata": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            # Adjusted metadata properties based on what's stored
                                                            "page_number": {"type": "string"},
                                                            "source_document_id": {"type": "string"},
                                                            "book_title": {"type": "string"}
                                                        }
                                                    }
                                                },
                                                "similarity_threshold_used": {"type": "number"},
                                                "filter_book_ids_used": {"type": "array", "items": {"type": "string"}, "nullable": True},
                                                "is_saved": {"type": "boolean"}
                                             }
                                        }
                                    }
                                }
                            }
                        },
                        # REMOVED 400 for missing anonymous_user_id
                        "401": {"description": "Unauthorized (missing or invalid token)"}, # <-- Added 401
                        "500": {"description": "Internal Server Error"},
                        "503": {"description": "Service Unavailable (System not initialized)"}
                    }
                }
            },
             # Add the /books endpoint definition for fetching study materials
             "/books": {
                 "get": {
                     "summary": "Retrieve list of study materials (Authentication Required).",
                     "security": [{"bearerAuth": []}], # <-- Requires bearer auth
                     "responses": {
                         "200": {
                             "description": "List of books.",
                             "content": {
                                 "application/json": {
                                     "schema": {
                                         "type": "array",
                                         "items": {
                                             "type": "object",
                                             "properties": {
                                                 "book_id": {"type": "string"},
                                                 "title": {"type": "string"},
                                                 "filepath": {"type": "string", "description": "Storage path or URL for the PDF."},
                                                 "processed": {"type": "boolean", "description": "True if embeddings have been generated."},
                                                 "subject": {"type": "string", "nullable": True},
                                                 "author": {"type": "string", "nullable": True},
                                                 "version": {"type": "string", "nullable": True},
                                                 "last_processed_at": {"type": "string", "format": "date-time", "nullable": True}
                                             }
                                         }
                                     }
                                 }
                             }
                         },
                         "401": {"description": "Unauthorized (missing or invalid token)"},
                         "500": {"description": "Internal Server Error"},
                         "503": {"description": "Supabase not initialized"}
                     }
                 }
             },
             # Add the /reprocess_book endpoint definition
            "/reprocess_book/{book_id_to_reprocess}": {
                "post": {
                    "summary": "Force reprocessing of a specific book (Authentication Required).", # Added auth requirement
                    "security": [{"bearerAuth": []}], # <-- Requires bearer auth
                    "parameters": [
                        {
                            "name": "book_id_to_reprocess", "in": "path", "required": True,
                            "description": "The book_id of the book to reprocess.",
                            "schema": {"type": "string"}
                        }
                    ],
                    "responses": {
                        "200": {"description": "Book reprocessing initiated successfully."},
                        "401": {"description": "Unauthorized (missing or invalid token)"}, # Added 401
                        "404": {"description": "Book not found in Supabase."},
                        "500": {"description": "Error during reprocessing."},
                        "503": {"description": "Service Unavailable (system not initialized)"}
                    }
                }
            }

        },
        # Define the security scheme for Bearer token
        "components": {
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT" # Standard format for JWT tokens
                }
            }
        }
    }
    return jsonify(spec)


# --- Add the new /books endpoint ---
@app_fi.route('/books', methods=['GET'])
def get_books_route():
    """Fetches the list of books from Supabase for authenticated users."""
    # --- Enforce Authentication ---
    user, auth_error = get_user_from_request()
    if auth_error:
        return jsonify({"error": auth_error}), 401 # 401 Unauthorized

    # Use the supabase client initialized in app.py
    if app.supabase is None:
         return jsonify({"error": "Backend Supabase client not configured."}), 503

    print(f"Authenticated user {user.id} requesting book list.")

    try:
        # Query the books table.
        # Assuming the books table is readable by the Service Role Key.
        # If books should be specific per user, add .eq('user_id', user.id) to the query filter.
        response = app.supabase.table('books')\
                     .select('book_id, title, filepath, processed, subject, author, version, last_processed_at')\
                     .order('title', { 'ascending': True })\
                     .execute() # Execute the query

        # Check for Supabase API errors during select
        if hasattr(response, 'error') and response.error:
             print(f"Supabase select books error for user {user.id}: {response.error.message}")
             traceback.print_exc()
             return jsonify({"error": response.error.message}), 500

        # Return the data as JSON
        return jsonify(response.data), 200

    except Exception as e:
        print(f"Error fetching books for user {user.id}: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal server error fetching books: {str(e)}"}), 500

# --- Add the /reprocess_book endpoint ---
@app_fi.route('/reprocess_book/<book_id_to_reprocess>', methods=['POST'])
def reprocess_single_book_route(book_id_to_reprocess):
    """Forces reprocessing of a specific book (Authentication Required)."""
     # --- Enforce Authentication ---
    user, auth_error = get_user_from_request()
    if auth_error:
        return jsonify({"error": auth_error}), 401 # 401 Unauthorized

    # Add an authorization check here if only specific users should reprocess
    # E.g., if user.id != 'admin_user_id': return jsonify({"error": "Permission denied"}), 403

    # Use the supabase client initialized in app.py
    if app.supabase is None:
        return jsonify({"error": "Backend Supabase client not configured."}), 503


    print(f"Authenticated user {user.id} attempting to reprocess book_id: {book_id_to_reprocess}")
    try:
        # Fetch book details from Supabase
        # Ensure book_id_to_reprocess matches the type in Supabase (string or UUID)
        response = app.supabase.table('books').select('book_id, title, filepath').eq('book_id', str(book_id_to_reprocess)).single().execute()

        if response.data:
            book_details = response.data
            book_title = book_details.get('title', book_id_to_reprocess)

            try:
                # Attempt to delete existing chunks for this book ID from Chroma
                print(f"Attempting to delete existing chunks for book '{book_title}' (ID: {book_id_to_reprocess}) from Chroma...")
                # Ensure Chroma collection is available
                if app.chroma_collection is None:
                     print("Error: Chroma collection not initialized. Cannot delete chunks.")
                     # Decide if this is a critical failure for reprocessing
                     return jsonify({"error": "ChromaDB not initialized."}), 503

                app.chroma_collection.delete(where={"source_document_id": str(book_id_to_reprocess)})
                print(f"Successfully deleted existing chunks for book '{book_title}' (ID: {book_id_to_reprocess}).")
            except Exception as e_delete:
                print(f"Warning: Could not delete all existing chunks for book '{book_title}' (ID: {book_id_to_reprocess}) from Chroma: {e_delete}")
                traceback.print_exc()
                # Decide if failure to delete should stop reprocessing - for now, just warn and continue


            # Generate and store new embeddings using the updated function that reads from Storage
            if app.embedding_model is None:
                 print("Error: Embedding model not initialized. Cannot generate embeddings.")
                 return jsonify({"error": "Embedding model not initialized."}), 503

            # generate_and_store_embeddings now uses extract_text_from_pdf which handles storage paths
            if app.generate_and_store_embeddings(book_details, app.chroma_collection, app.embedding_model):
                # Update 'processed' status and timestamp in Supabase DB
                print(f"Successfully reprocessed and stored embeddings for: '{book_title}'")
                try:
                    app.supabase.table('books').update({
                        'processed': True,
                        'last_processed_at': datetime.datetime.utcnow().isoformat()
                    }).eq('book_id', str(book_id_to_reprocess)).execute()
                    print(f"Updated 'processed' status for book '{book_id_to_reprocess}' in Supabase.")
                    return jsonify({"message": f"Book '{book_title}' reprocessed successfully."}), 200
                except Exception as e_supa:
                    print(f"Error updating Supabase status after reprocessing for book '{book_id_to_reprocess}': {e_supa}")
                    traceback.print_exc()
                    return jsonify({"error": f"Failed to update status in DB after reprocessing '{book_title}'."}), 500
            else:
                # If generate_and_store_embeddings returned False
                print(f"Failed to reprocess book: '{book_title}' (ID: {book_id_to_reprocess}). Check logs for extraction/embedding errors.")
                return jsonify({"error": f"Failed to reprocess book '{book_title}'."}), 500
        else:
            # If book not found in DB
            print(f"Supabase query response for book_id '{book_id_to_reprocess}': Book not found.")
            return jsonify({"error": f"Book with book_id '{book_id_to_reprocess}' not found in Supabase."}), 404

    except Exception as e:
        print(f"Error during reprocessing book {book_id_to_reprocess} for user {user.id}: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal server error during reprocessing: {str(e)}"}), 500


# --- Modify existing endpoints to use get_user_from_request ---

@app_fi.route('/ask', methods=['POST'])
def ask_question_route():
    """Answers a question based on document content (Authentication Required)."""
    # --- Enforce Authentication ---
    user, auth_error = get_user_from_request() # <--- Get user
    if auth_error:
        return jsonify({"error": auth_error}), 401 # <--- Enforce Auth

    # User is authenticated, get their ID
    user_id = user.id
    print(f"Authenticated user {user_id} asking question.")

    # Check if core RAG components are initialized
    if app.chroma_collection is None or app.embedding_model is None:
        return jsonify({"error": "System not initialized. Please wait or check logs."}), 503

    data = request.get_json()
    # Check for required 'question' field
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    question = data['question']
    # Use user_id obtained from the token, NOT from request body
    # anonymous_user_id = data['anonymous_user_id'] # REMOVE THIS LINE

    top_n = data.get('top_n_chunks', 5)
    similarity_threshold = data.get('similarity_threshold', 0.5)
    filter_book_ids = data.get('filter_book_ids', None)

    # Construct the where filter for ChromaDB if book IDs are provided
    where_filter = None
    if filter_book_ids and isinstance(filter_book_ids, list) and len(filter_book_ids) > 0:
        # Ensure book_ids are strings for consistent filtering in Chroma metadata
        str_filter_book_ids = [str(bid) for bid in filter_book_ids]
        if len(str_filter_book_ids) == 1:
            where_filter = {"source_document_id": str_filter_book_ids[0]}
        else:
            where_filter = {"source_document_id": {"$in": str_filter_book_ids}}


    # Find relevant chunks using the Chroma collection and embedding model initialized in app.py
    relevant_chunks = app.find_relevant_chunks_chroma(
        question, app.chroma_collection, app.embedding_model, top_n, similarity_threshold, where_filter
    )

    # Get the AI answer using the relevant chunks
    ai_answer = app.get_answer_from_openai(question, relevant_chunks)

    # --- Store interaction in Supabase DB ---
    interaction_id = str(uuid.uuid4()) # Generate a UUID for the interaction
    try:
        # Prepare metadata from relevant chunks for storage
        chunks_metadata = [
            {
                'page_number': c.get('page_number', 'N/A'),
                'source_document_id': c.get('source_document_id', 'N/A'),
                'book_title': c.get('book_title', 'N/A')
            } for c in relevant_chunks
        ]
        # Filter out chunks with essential missing metadata if desired
        chunks_metadata = [m for m in chunks_metadata if m.get('source_document_id') not in [None, '', 'N/A']] # Basic check

        insert_data = {
            'id': interaction_id,
            'user_id': user_id, # <--- Use the validated user ID (UUID)
            'question_text': question,
            'answer_text': ai_answer,
            'relevant_chunks_metadata': chunks_metadata,
            'similarity_threshold_used': similarity_threshold,
            'filter_book_ids_used': filter_book_ids # Store the original list or None
        }

        # Use the supabase client initialized in app.py to insert data
        if app.supabase is None:
             print("Error: Supabase client not available for interaction logging.")
             # Decide if interaction logging failure should stop the response - usually not
             pass # Log and continue

        else:
            response = app.supabase.table('interactions').insert(insert_data).execute()

            # Check Supabase response for errors during insert
            if hasattr(response, 'error') and response.error:
                 print(f"Supabase insert error for user {user_id}, interaction {interaction_id}: {response.error.message}")
                 traceback.print_exc()
                 # Decide on error handling - maybe log and return response anyway?
                 pass # Log the error and return the answer to the user

    except Exception as e:
        print(f"Error storing interaction for user {user_id}: {e}")
        traceback.print_exc()
        # Decide on error handling - maybe log and return response anyway?
        pass # Log the error and return the answer to the user


    # Return the answer and relevant chunks to the frontend
    return jsonify({
        "interaction_id": interaction_id, # Return the interaction ID so frontend can save it
        "question": question,
        "answer": ai_answer,
        "relevant_chunks": relevant_chunks # Include sources in the response
    })

@app_fi.route('/save_interaction', methods=['POST'])
def save_interaction_route():
    """Marks an interaction as saved or unsaved for the authenticated user (Authentication Required)."""
    # --- Enforce Authentication ---
    user, auth_error = get_user_from_request() # <--- Get user
    if auth_error:
        return jsonify({"error": auth_error}), 401 # <--- Enforce Auth

    # User is authenticated, get their ID
    user_id = user.id
    print(f"Authenticated user {user_id} attempting to save interaction.")

    # Use the supabase client initialized in app.py
    if app.supabase is None:
        return jsonify({"error": "Backend Supabase client not configured."}), 503

    data = request.get_json()
    # Check for required fields in the request body
    if not data or not all(k in data for k in ['interaction_id', 'is_saved']):
        return jsonify({"error": "Missing required data (interaction_id, is_saved)."}), 400

    interaction_id = data['interaction_id']
    is_saved = bool(data['is_saved']) # Ensure is_saved is boolean

    try:
        # Use the validated user_id in the update query to ensure user owns the interaction
        result = app.supabase.table('interactions').update({'is_saved': is_saved})\
            .eq('id', interaction_id)\
            .eq('user_id', user_id)\
            .execute()

        # Check for Supabase API errors during update
        if hasattr(result, 'error') and result.error:
            print(f"Supabase update error for user {user_id}, interaction {interaction_id}: {result.error.message}")
            traceback.print_exc()
            # Check if it's a "Not Found" error (PGRST116) which implies the ID/User combo didn't exist
            if result.error.code == 'PGRST116':
                 return jsonify({"error": "Interaction not found for this user."}), 404
            # Other database errors
            return jsonify({"error": result.error.message}), 500

        # Check if the update actually affected any rows.
        # Supabase client might return data=[] if no rows matched the .eq filters.
        if result.data and len(result.data) > 0:
             print(f"Successfully updated save status for interaction {interaction_id} for user {user_id} to {is_saved}.")
             return jsonify({"message": "Save status updated."}), 200
        else:
             # If no error, but no data, it means the interaction ID existed but didn't match the user_id
             print(f"Update query for interaction {interaction_id}, user {user_id} executed but no row affected (ID/User mismatch?).")
             return jsonify({"error": "Interaction not found for this user."}), 404 # Return 404 if no row affected


    except Exception as e:
        print(f"Error during save_interaction for user {user_id}, interaction {interaction_id}: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app_fi.route('/saved_interactions', methods=['GET'])
def get_saved_interactions_route():
    """Retrieves saved interactions for the authenticated user (Authentication Required)."""
    # --- Enforce Authentication ---
    user, auth_error = get_user_from_request() # <--- Get user
    if auth_error:
        # Accessing saved interactions requires authentication
        return jsonify({"error": auth_error}), 401 # 401 Unauthorized

    # User is authenticated, get their ID
    user_id = user.id
    print(f"Authenticated user {user_id} requesting saved interactions.")

    # Use the supabase client initialized in app.py
    if app.supabase is None:
        return jsonify({"error": "Backend Supabase client not configured."}), 503

    # REMOVED: Getting anonymous_user_id from query params
    # anonymous_user_id = request.args.get('anonymous_user_id') # REMOVE THIS LINE

    try:
        # Query the interactions table filtered by the authenticated user_id and is_saved = True
        result = app.supabase.table('interactions')\
            .select("*")\
            .eq('user_id', user_id)\
            .eq('is_saved', True)\
            .order('created_at', desc=True)\
            .execute()

        # Check for Supabase API errors during select
        if hasattr(result, 'error') and result.error:
            print(f"Supabase select saved interactions error for user {user_id}: {result.error.message}")
            traceback.print_exc()
            return jsonify({"error": result.error.message}), 500

        # Return the list of interactions (can be empty)
        return jsonify(result.data), 200

    except Exception as e:
        print(f"Error fetching saved interactions for user {user_id}: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# --- Optional Index Route ---
@app_fi.route('/')
def index():
    """Simple index route."""
    return jsonify({"message": "TNPSC Q&A API (Authenticated) running. See /api/docs for Swagger UI."})


# --- Main Execution for fi.py ---
# This block runs when you execute `python fi.py`
if __name__ == '__main__':
    print("--- Starting fi.py system initialization ---")
    # Call the initialization function from the imported app module
    # This initializes models, connects to Chroma/Supabase, and processes new books
    if app.initialize_system():
        # Check if the critical components were successfully initialized in app.py
        # The initialize_system function now returns True only if core components are ready
        if app.chroma_collection and app.embedding_model and app.supabase:
            print("--- fi.py system initialization complete. Starting Flask server ---")
            # You might want to use a production WSGI server like Gunicorn in deployment
            # For local testing, app_fi.run(debug=True) is fine
            print(f"Flask application running on http://0.0.0.0:5001")
            print(f"Swagger API docs available at http://0.0.0.0:5001{SWAGGER_URL_FI}")
            # Run the Flask app defined *in fi.py*
            # For production, you'd use 'gunicorn -w 4 fi:app_fi -b 0.0.0.0:5001' or similar
            app_fi.run(debug=True, port=5001, host='0.0.0.0')
        else:
            # Initialization failed (e.g., failed to load model, connect to DB/Chroma)
            print("--- fi.py system initialization failed. Flask server will not start ---")
            # app.initialize_system should have printed specific errors

    else:
        # Critical failure during app.initialize_system (e.g., Supabase client failed to init)
        print("--- Critical failure during fi.py system initialization. Flask server will not start ---")
        # app.initialize_system should have printed specific errors

# --- END OF FILE fi.py (Updated) ---