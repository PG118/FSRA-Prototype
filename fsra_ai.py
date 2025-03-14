

import openai
import json
import numpy as np
import streamlit as st
import os

# Load OpenAI API key (Replace with your actual key)
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load preprocessed regulations with embeddings
with open("regulations_with_embeddings.json", "r", encoding="utf-8") as f:
    regulations = json.load(f)

# File for storing past decisions
decisions_file = "policy_decisions.json"

# Ensure the decision storage file exists
if not os.path.exists(decisions_file):
    with open(decisions_file, "w", encoding="utf-8") as f:
        json.dump([], f)  # Create an empty JSON file

# Load past decisions
with open(decisions_file, "r", encoding="utf-8") as f:
    past_decisions = json.load(f)

# Function to generate embeddings for new policy change requests
def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# Function to find the most relevant regulation using cosine similarity
def find_relevant_regulation(policy_change):
    policy_embedding = get_embedding(policy_change)

    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    best_match = None
    best_score = -1

    for reg in regulations:
        score = cosine_similarity(policy_embedding, np.array(reg["embedding"]))
        if score > best_score:
            best_score = score
            best_match = reg

    return best_match

# Function to save AI decisions
def save_decision(policy_request, decision, feedback=""):
    entry = {"request": policy_request, "decision": decision, "feedback": feedback}
    past_decisions.append(entry)
    
    with open(decisions_file, "w", encoding="utf-8") as f:
        json.dump(past_decisions, f, indent=4)

# Function to check past decisions before running AI again
def find_past_decision(policy_request):
    for entry in past_decisions:
        if entry["request"].strip().lower() == policy_request.strip().lower():
            return entry["decision"]
    return None

# Function to analyze policy change
def analyze_policy_change(policy_change):
    # Check if a similar request was made before
    past_decision = find_past_decision(policy_change)
    if past_decision:
        return f"(Retrieved from past decisions) {past_decision}"

    # Retrieve the most relevant regulation
    relevant_regulation = find_relevant_regulation(policy_change)

    if not relevant_regulation:
        return "No relevant regulation found. Manual review required."

    # Send to GPT-4 for approval decision
    prompt = f"""
    Given the following insurance regulation:

    {relevant_regulation["title"]}: {relevant_regulation["description"]}

    Evaluate the following policy change request:

    "{policy_change}"

    Should this be approved or rejected? Provide justification.
    """

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a regulatory AI assistant."},
                  {"role": "user", "content": prompt}]
    )

    decision = response.choices[0].message.content
    save_decision(policy_change, decision)  # Store decision for future reference
    return decision

# Initialize session state for decision storage
if "decision" not in st.session_state:
    st.session_state.decision = ""

# Title
st.title("Ontario Auto Insurance AI Policy Reviewer")

# User inputs policy change request
policy_change = st.text_area("Enter the proposed policy change:")

if st.button("Analyze Policy Change"):
    decision = analyze_policy_change(policy_change)
    st.session_state.decision = decision  # Store decision in session state
    st.write(f"**Decision:** {st.session_state.decision}")

# Display the AI decision if it's already generated
if st.session_state.decision:
    st.write(f"**Decision:** {st.session_state.decision}")

    # Allow user to modify the decision
    additional_info = st.text_area("Modify or add additional details:")
    
    if st.button("Update Decision"):
        updated_request = policy_change + "\n\nAdditional Information: " + additional_info
        
        # Generate updated AI response
        updated_decision = analyze_policy_change(updated_request)
        
        # Save the updated decision and keep it in session state
        save_decision(updated_request, updated_decision, additional_info)
        st.session_state.decision = updated_decision  # Update session state
        
        st.write(f"**Updated Decision:** {st.session_state.decision}")
