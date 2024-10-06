<img src='https://github.com/fabiomatricardi/OpenVINO-StableLM-3B-streamlit/blob/main/images/banner.png' width=900>

# OpenVINO-StableLM-3B-streamlit
A streamlit ChatBot running StableLM Zephyr 3B with Openvino and Optimum Intel

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


