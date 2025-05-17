from supabase import create_client, Client

SUPABASE_URL = "https://lztqivttzdgumataatph.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx6dHFpdnR0emRndW1hdGFhdHBoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDY3OTgwNjYsImV4cCI6MjA2MjM3NDA2Nn0.2_7Zy84trjEq3Rm5Uy9v-nLoTOFbc8JCuNRALPhQ6dw" # Use Anon key for client-side auth

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Sign in a test user
email = "shafiudeenmd@gmail.com"
password = "Shafi@1"

try:
    response = supabase.auth.sign_in_with_password({"email": email, "password": password})
    if response.session and response.session.access_token:
        jwt_token = response.session.access_token
        print(f"JWT Token: {jwt_token}")
    else:
        print("Login failed or no access token in response.")
        if response.error:
            print(f"Error: {response.error.message}")
except Exception as e:
    print(f"An error occurred: {e}")