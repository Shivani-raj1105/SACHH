import streamlit as st

st.write("Minimal Feedback Button Test")

# Step 1: Show feedback message if present
if 'feedback_message' in st.session_state:
    st.success(st.session_state['feedback_message'])
    del st.session_state['feedback_message']

# Step 2: User enters text and clicks 'Submit'
input_text = st.text_area("Paste news headline or article:", height=200)
if 'submitted_text' not in st.session_state:
    st.session_state['submitted_text'] = ""
if st.button("Submit"):
    st.session_state['submitted_text'] = input_text

# Step 3: Show feedback buttons for the submitted text
submitted_text = st.session_state['submitted_text']
stable_input_key = submitted_text.strip().lower()
if 'feedback_given' not in st.session_state:
    st.session_state['feedback_given'] = {}

disable_buttons = st.session_state['feedback_given'].get(stable_input_key, False)

st.write(f"Submitted text: {submitted_text}")

col1, col2, col3 = st.columns(3)
clicked = None
with col1:
    if st.button("Flag as Correct", key=f"correct_{stable_input_key}", disabled=disable_buttons):
        clicked = "correct"
with col2:
    if st.button("Flag as Incorrect", key=f"incorrect_{stable_input_key}", disabled=disable_buttons):
        clicked = "incorrect"
with col3:
    if st.button("Not a News", key=f"notnews_{stable_input_key}", disabled=disable_buttons):
        clicked = "notnews"

if clicked:
    st.session_state['feedback_given'][stable_input_key] = True
    st.session_state['feedback_message'] = f"!!! {clicked.capitalize()} Button block entered !!!"
    print(f"!!! {clicked.capitalize()} Button block entered !!!")
    st.rerun()