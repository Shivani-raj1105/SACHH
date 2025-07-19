import streamlit as st
import requests
import json
import base64
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

def verify_google_token(token):
    try:
        idinfo = id_token.verify_oauth2_token(
            token, 
            google_requests.Request(), 
            st.secrets["google_oauth"]["client_id"]
        )
        st.write("Decoded id_token:", idinfo)
        return idinfo
    except Exception as e:
        st.error(f"Token verification failed: {e}")
        print("Token verification failed:", e)
        return None

def get_google_oauth_url():
    client_id = st.secrets["google_oauth"]["client_id"]
    redirect_uri = "http://localhost:8501"
    scope = "openid email profile"
    
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth"
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": scope,
        "access_type": "offline",
        "prompt": "consent"
    }
    
    param_string = "&".join([f"{k}={v}" for k, v in params.items()])
    return f"{auth_url}?{param_string}"

def handle_google_oauth_callback():
    # Only show error if a login attempt was made
    login_attempted = False
    if 'code' in st.query_params:
        login_attempted = True
        code = st.query_params['code']
        try:
            token_url = "https://oauth2.googleapis.com/token"
            data = {
                "client_id": st.secrets["google_oauth"]["client_id"],
                "client_secret": st.secrets["google_oauth"]["client_secret"],
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": "http://localhost:8501"
            }
            response = requests.post(token_url, data=data)
            token_data = response.json()
            if 'id_token' in token_data:
                user_info = verify_google_token(token_data['id_token'])
                if user_info:
                    st.session_state.user_info = {
                        "name": user_info.get('name', 'User'),
                        "email": user_info.get('email', ''),
                        "picture": user_info.get('picture', ''),
                        "sub": user_info.get('sub', '')
                    }
                    st.success("Successfully signed in!")
                    st.query_params.clear()  # <-- Add this line
                    st.rerun()
                    return
                else:
                    if login_attempted:
                        st.error("Failed to get user information")
            else:
                if login_attempted:
                    st.error("Failed to get user information")
        except Exception as e:
            if login_attempted:
                st.error(f"Failed to get user information: {e}")

def login_section():
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
    
    handle_google_oauth_callback()
    
    if st.session_state.user_info is None:
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background: #222; border-radius: 10px; margin: 2rem 0;'>
            <h2 style='color: #FFD700;'>Welcome to SACH</h2>
            <p style='color: #e5c07b;'>Please sign in to access the fake news detector</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Sign in with Google", type="primary", use_container_width=True):
                auth_url = get_google_oauth_url()
                st.markdown(f"""
                <div style='text-align: center; padding: 1rem; background: #333; border-radius: 8px; margin: 1rem 0;'>
                    <p style='color: #e5c07b;'>Click the link below to sign in with Google:</p>
                    <a href="{auth_url}" target="_blank" style='color: #FFD700; text-decoration: none; font-weight: bold;'>
                        Sign in with Google
                    </a>
                </div>
                """, unsafe_allow_html=True)
        
        st.info("For demo purposes, you can also proceed as a guest:")
        if st.button("Continue as Guest"):
            st.session_state.user_info = {"name": "Guest User", "email": "guest@example.com"}
            st.rerun()
        
        st.stop()
    else:
        user_info = st.session_state.user_info
        st.sidebar.markdown(f"""
        <div style='background: #FFD700; color: #000; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
            <strong>Welcome, {user_info.get('name', 'User')}!</strong><br>
            <small>{user_info.get('email', '')}</small>
        </div>
        """, unsafe_allow_html=True)
        
        if st.sidebar.button("Sign Out"):
            st.session_state.user_info = None
            st.rerun()
        return  # <-- Ensure function exits after successful login

def require_auth():
    if 'user_info' not in st.session_state or st.session_state.user_info is None:
        st.error("Please sign in to access this feature.")
        st.stop()
    return st.session_state.user_info 