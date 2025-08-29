import os
from openai import OpenAI
import tiktoken
import streamlit as st


def _secret(key: str):
    try:
        return st.secrets[key]
    except Exception:
        return None

# Prefer VENICE_API_KEY, fall back to OPENAI_API_KEY for compatibility
api_key = (
    _secret("VENICE_API_KEY")
    or os.getenv("VENICE_API_KEY")
    or _secret("OPENAI_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)

BASE_URL = (
    _secret("VENICE_BASE_URL")
    or os.getenv("VENICE_BASE_URL")
    or "https://api.venice.ai/api/v1"
)

if not api_key:
    st.error("Missing API key. Set VENICE_API_KEY (preferred) or OPENAI_API_KEY in Streamlit secrets or environment.")
    st.stop()

client = OpenAI(api_key=api_key, base_url=BASE_URL)
MODEL = "llama-3.3-70b"
TEMPERATURE = 0.7
MAX_TOKENS = 100
TOKEN_BUDGET = 1000
SYSTEM_PROMPT = "You are a fed up and sassy assistant who hates answering questions."

def get_encoding(model):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"Warning: Tokenizer for model '{model}' not found. Falling back to 'cl100k_base'.")
        return tiktoken.get_encoding("cl100k_base")

ENCODING = get_encoding(MODEL)

def count_tokens(text):
    return len(ENCODING.encode(text))

def total_tokens_used(messages):
    try:
        return sum(count_tokens(msg["content"]) for msg in messages)
    except Exception as e:
        print(f"[token count error]: {e}")
        return 0

def enforce_token_budget(messages, budget=TOKEN_BUDGET):
    try:
        while total_tokens_used(messages) > budget:
            if len(messages) <= 2:
                break 
            messages.pop(1)
    except Exception as e:
        print(f"[token budget error]: {e}")


def chat(user_input, model, temperature=TEMPERATURE, max_tokens=MAX_TOKENS):
    messages = st.session_state.messages
    messages.append({"role": "user", "content": user_input})

    enforce_token_budget(messages)

    venice_params = {
        "include_venice_system_prompt": bool(include_venice_sys),
        "enable_web_search": "on" if bool(use_web_search) else "off",
    }

    with st.spinner("Thinking..."):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"venice_parameters": venice_params},
        )
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    return reply

### Streamlit ###

st.title("Sassy Chatbot")
st.sidebar.header("Options")
st.sidebar.write("This is a demo of a sassy chatbot using Venice AI's OpenAI-compatible API.")

# Model + Venice options
model_id = st.sidebar.text_input("Model ID", value=MODEL, help="Example: llama-3.3-70b, qwen2.5-coder-32b, etc.")
include_venice_sys = st.sidebar.checkbox("Include Venice system prompt", value=False,
    help="If enabled, Venice may prepend its own system instructions.")
use_web_search = st.sidebar.checkbox("Enable Venice web search", value=False)

max_tokens = st.sidebar.slider("Max Tokens", 1, 250, 100)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
system_message_type = st.sidebar.selectbox("System Message", ("Sassy Assistant", "Angry Assistant", "Custom"))

if system_message_type == "Sassy Assistant":
    SYSTEM_PROMPT = "You are a sassy assistant that is fed up with answering questions."
elif system_message_type == "Angry Assistant":
    SYSTEM_PROMPT = "You are an angry assistant that likes yelling in all caps."
elif system_message_type == "Custom":
    SYSTEM_PROMPT = st.sidebar.text_area("Custom System Message", "Enter your custom system message here.")
else:
    SYSTEM_PROMPT = "You are a helpful assistant."

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

if st.sidebar.button("Apply New System Message"):
    st.session_state.messages[0] = {"role": "system", "content": SYSTEM_PROMPT}
    st.success("System message updated.")

if st.sidebar.button("Reset Conversation"):
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    st.success("Conversation reset.")

if prompt := st.chat_input("What is up?"):
    reply = chat(prompt, model=model_id.strip() or MODEL, temperature=temperature, max_tokens=max_tokens)

for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])