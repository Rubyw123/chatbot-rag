#llama3 prompt
llama3_prompt_msg = """
"<|start_header_id|>system<|end_header_id|>

You are an AI chatbot having a conversation with a user. Answer the user's question.
<|eot_id|><|start_header_id|>user<|end_header_id|>

history:{history}
question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""



