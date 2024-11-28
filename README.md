<img src='https://github.com/fabiomatricardi/OpenVINO-StableLM-3B-streamlit/blob/main/images/banner.png' width=900>

# OpenVINO-StableLM-3B-streamlit [![Mentioned in Awesome OpenVINO](https://awesome.re/mentioned-badge-flat.svg)](https://github.com/openvinotoolkit/awesome-openvino)
A streamlit ChatBot running StableLM Zephyr 3B with Openvino and Optimum Intel

---

* [StableLM-3B Chatbot](https://github.com/fabiomatricardi/OpenVINO-StableLM-3B-streamlit) - A streamlit CHATBOT interface with [stablelm-zephyr-3b](https://huggingface.co/stabilityai/stablelm-zephyr-3b) quantized in 4bit and optimum-intel. The Interface has a kind text streaming effect, and the number of turns are handled to not exceed the context window. The Model used is [published on Hugging Face Hub](https://huggingface.co/FM-1976/stablelm-zephyr-3b-openvino-4bit) and was created with the free HF Space hosting the [NCCF-quantization tool](https://huggingface.co/spaces/OpenVINO/nncf-quantization).

---

## Requirements
```
python312 -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install openvino-genai==2024.4.0
pip install optimum-intel[openvino] tiktoken streamlit==1.36.0
```

## Download the model
from HuggingFace 

[https://huggingface.co/FM-1976/stablelm-zephyr-3b-openvino-4bit](https://huggingface.co/FM-1976/stablelm-zephyr-3b-openvino-4bit)


## Run the app
```
streamlit run .\stappStableLM.py
```


