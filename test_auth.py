import streamlit as st
from auth import login_section, require_auth

st.set_page_config(
    page_title="Auth Test",
    page_icon="",
    layout="wide"
)

st.title("Authentication Test")

login_section()

if st.session_state.get('user_info'):
    st.success(f"✅ Logged in as: {st.session_state.user_info.get('name', 'User')}")
    st.info(f"Email: {st.session_state.user_info.get('email', 'No email')}")
    
    if st.button("Test Protected Feature"):
        user = require_auth()
        st.success(f"Protected feature accessed by: {user.get('name', 'User')}")
else:
    st.warning("Please log in to access the app") 