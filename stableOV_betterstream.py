### source code https://github.com/yas-sim/openvino-llm-minimal-code/blob/main/inference-stream.py

from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer, AutoConfig
from threading import Thread
from transformers import TextIteratorStreamer
import warnings
warnings.filterwarnings(action='ignore')
import sys
import datetime
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base") #context_count = len(encoding.encode(yourtext))

def countTokens(text):
    encoding = tiktoken.get_encoding("cl100k_base") #context_count = len(encoding.encode(yourtext))
    numoftokens = len(encoding.encode(text))
    return numoftokens

def printStats(delta,question,response):
    totalseconds = delta.total_seconds()
    print('\n---')
    print(f'Inference time: {delta}')
    prompttokens = countTokens(question)
    assistanttokens = countTokens(response)
    totaltokens = prompttokens + assistanttokens
    speed = totaltokens/totalseconds
    genspeed = assistanttokens/totalseconds
    print(f"Prompt Tokens: {prompttokens}")
    print(f"Output Tokens: {assistanttokens}")
    print(f"TOTAL Tokens: {totaltokens}")
    print('---')
    print(f'>>>Inference speed: {speed:.3f}  t/s')
    print(f'>>>Generation speed: {genspeed:.3f}  t/s\n\n')
    print('---')

model_id = 'stablelm-zephyr-3b-openvino-4bit' #from my HF repo
model_precision = ['FP16', 'INT8', 'INT4', 'INT4_stateless'][2]
print(f'LLM model: stablelm-zephyr-3b-openvino-4bit, PRECISION: {model_precision}')

tokenizer = AutoTokenizer.from_pretrained(model_id)
ov_model = OVModelForCausalLM.from_pretrained(
    model_id = model_id,
    device='CPU',
    ov_config={"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""},
    config=AutoConfig.from_pretrained(model_id)
)

# Generation with a prompt message
question = 'Explain the plot of Cinderella in a sentence.'
messages = [
    {"role": "user", "content": question}
]

print('Question:', question)
#Credit to https://github.com/openvino-dev-samples/chatglm3.openvino/blob/main/chat.py
streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
model_inputs = tokenizer.apply_chat_template(messages,
                                                     add_generation_prompt=True,
                                                     tokenize=True,
                                                     pad_token_id=tokenizer.eos_token_id,
                                                     num_return_sequences=1,
                                                     return_tensors="pt")
generate_kwargs = dict(input_ids=model_inputs,
                        max_new_tokens=450,
                        temperature=0.1,
                        do_sample=True,
                        top_p=0.5,
                        repetition_penalty=1.178,
                        streamer=streamer)
t1 = Thread(target=ov_model.generate, kwargs=generate_kwargs)
t1.start()
start = datetime.datetime.now()
partial_text = ""
for new_text in streamer:
    new_text = new_text
    print(new_text, end="", flush=True)
    partial_text += new_text
response = partial_text
delta = datetime.datetime.now() - start
printStats(delta,question,response)

while True:
    userinput = ""
    print("\033[1;30m")  #dark grey
    print("Enter your text (end input with Ctrl+D on Unix or Ctrl+Z on Windows) - type quit! to exit the chatroom:")
    print("\033[91;1m")  #red
    lines = sys.stdin.readlines()
    for line in lines:
        userinput += line + "\n"
    if "quit!" in lines[0].lower():
        print("\033[0mBYE BYE!")
        break
    messages = [{"role": "user", "content": userinput}]
    print("\033[92;1m") 
    streamer = TextIteratorStreamer(tokenizer, timeout=180.0, skip_prompt=True, skip_special_tokens=True)
    model_inputs = tokenizer.apply_chat_template(messages,
                                                        add_generation_prompt=True,
                                                        tokenize=True,
                                                        return_tensors="pt")
    generate_kwargs = dict(input_ids=model_inputs,
                            max_new_tokens=450,
                            temperature=0.1,
                            do_sample=True,
                            top_p=0.5,
                            repetition_penalty=1.178,
                            streamer=streamer)
    t1 = Thread(target=ov_model.generate, kwargs=generate_kwargs)
    t1.start()
    start = datetime.datetime.now()
    partial_text = ""
    for new_text in streamer:
        new_text = new_text
        print(new_text, end="", flush=True)
        partial_text += new_text
    response = partial_text
    delta = datetime.datetime.now() - start
    printStats(delta,userinput,response)