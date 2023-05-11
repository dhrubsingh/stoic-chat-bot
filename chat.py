import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from streamlit_chat import message


model = GPT2LMHeadModel.from_pretrained('meditations_model')
tokenizer = GPT2Tokenizer.from_pretrained('meditations_model')

# Define chatbot function
def chatbot(text):
    # Replace [MASK] token with user input
    text = text.replace("", "")
    text = text + " [MASK]"
    
    # Tokenize input
    input_ids = tokenizer.encode(text, return_tensors='pt')
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

    # Get predictions
    output = model(input_ids)[0]
    mask_token_logits = output[0, mask_token_index, :]
    top_token = torch.argmax(mask_token_logits, dim=1).tolist()[0]

    # Replace [MASK] with predicted token
    sequence = input_ids.tolist()[0]
    sequence[mask_token_index.tolist()[0]] = top_token
    response = tokenizer.decode(sequence, skip_special_tokens=True)
    
    return response

# Define Streamlit app
st.set_page_config(page_title="Stoic Chatbot", page_icon=":guardsman:", layout="wide")
st.title("Stoic Chatbot")

st.markdown("---")

chat_history = []

user_input = st.text_input("You:", value="", max_chars=200, key="input")
if user_input:
    message(user_input, is_user=True)
    response = chatbot(user_input)
    message(response)
    chat_history.append({"text": user_input, "from_user": True})
    chat_history.append({"text": response, "from_user": False})

st.markdown("---")

st.sidebar.subheader("About")
st.sidebar.markdown("This is a simple chatbot that uses ALBERT to generate responses to user input based on the text of Meditations by Marcus Aurelius. The model was fine-tuned on a subset of the text to improve the quality of the responses.")

st.sidebar.subheader("Instructions")
st.sidebar.markdown("Enter your message in the text box and press enter to generate a response. The chatbot will use ALBERT to generate a response based on the text of Meditations.")

st.sidebar.subheader("Feedback")
st.sidebar.markdown("If you have any feedback or suggestions for how to improve the chatbot, please feel free to leave a comment below or [contact the author](https://github.com/yourusername/yourproject).")
