import os
from queue import Queue
from threading import Thread
import textwrap

import gradio as gr
from transformers import LlamaForCausalLM, LlamaTokenizer


class StreamHandler:
    def __init__(self):
        self.queue = Queue()

    def put(self, item):
        self.queue.put({"type": "content", "content": item}, block=False)

    def end(self):
        self.queue.put({"type": "termination", "content": None}, block=False)


def format_prompt(history, message, system_prompt):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    prompt = f"{B_INST} {B_SYS}{system_prompt}{E_SYS} "

    for user_msg, asst_msg in history:
        user_msg = str(user_msg).strip()
        asst_msg = str(asst_msg).strip()

        prompt += f"{user_msg} {E_INST} {asst_msg} </s><s> {B_INST} "

    message = str(message).strip()
    prompt += f"{message} {E_INST} "

    return prompt


def build_generator(
    model_name,
    auth_token,
    temperature=0.6,
    top_p=0.9,
    max_gen_len=4096,
):
    SYSTEM_PROMPT = """\
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
    SYSTEM_PROMPT = textwrap.dedent(SYSTEM_PROMPT).strip()

    model = LlamaForCausalLM.from_pretrained(
        model_name, token=auth_token, load_in_4bit=True, device_map="auto"
    ).eval()
    tokenizer = LlamaTokenizer.from_pretrained(model_name, token=auth_token)

    # Alternative implementation using streaming
    def generate_process(inputs, stream_handler):
        model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            streamer=stream_handler,
        )

    def stream_response(message, history):
        prompt = format_prompt(history, message, SYSTEM_PROMPT)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        stream_handler = StreamHandler()

        t = Thread(target=generate_process, args=(inputs, stream_handler))
        t.start()

        # The first item in the queue contains the content, so we can ignore it
        stream_handler.queue.get(block=True)

        # Start now
        token_ids = []
        while True:
            item = stream_handler.queue.get(block=True)
            if item["type"] == "termination":
                break

            token_id = item["content"][0].item()
            token_ids.append(token_id)
            yield tokenizer.decode(token_ids, skip_special_tokens=True)

        # Wait for the thread to finish
        t.join()

    return stream_response


if __name__ == "__main__":
    print("Building generator...")
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    auth_token = os.environ["HUGGINGFACE_TOKEN"]
    respond = build_generator(model_name=model_name, auth_token=auth_token)

    print("Starting server...")
    title = model_name.split("/")[-1].replace("-", " ").title()
    desc = f"This Space demonstrates [{model_name}](https://huggingface.co/{model_name}) by Meta."
    css = """.toast-wrap { display: none !important } """
    ci = gr.ChatInterface(respond, title=title, description=desc, css=css)
    ci.queue()
    ci.launch()
