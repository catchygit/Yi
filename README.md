<p align="left">
    &nbspEnglish&nbsp | &nbsp; <a href="README_CN.md">中文</a>
</p>
<br><br>

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/01-ai/Yi/main/assets/img/Yi_logo_icon_dark.svg" width="100px">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/01-ai/Yi/main/assets/img/Yi_logo_icon_light.svg" width="100px"> 
  <img alt="specify theme context for images" src="https://raw.githubusercontent.com/01-ai/Yi/main/assets/img/Yi_logo_icon_light.svg" width="100px">
</picture>

</br>
</br>

<a href="https://github.com/01-ai/Yi/actions/workflows/build_docker_image.yml">
  <img src="https://github.com/01-ai/Yi/actions/workflows/build_docker_image.yml/badge.svg">
</a>
<a href="https://github.com/01-ai/Yi/blob/main/LICENSE">
  <img src="https://img.shields.io/badge/Code_License-Apache_2.0-lightblue">
</a>
<a href="https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt">
  <img src="https://img.shields.io/badge/Model_License-Yi_License-lightblue">
</a>
<a href="mailto:oss@01.ai">
  <img src="https://img.shields.io/badge/✉️-yi@01.ai-FFE01B">
</a>

</div>

<div id="top"></div>  

<div align="center">
  <h3 align="center">Building the Next Generation of Open-Source and Bilingual LLMs</h3>
</div>

<p align="center">
🤗 <a href="https://huggingface.co/01-ai" target="_blank">Hugging Face</a> • 🤖 <a href="https://www.modelscope.cn/organization/01ai/" target="_blank">ModelScope</a> • ✡️ <a href="https://wisemodel.cn/organization/01.AI" target="_blank">WiseModel</a>
</p> 

<p align="center">
    👩‍🚀 Ask questions or discuss ideas on <a href="https://github.com/01-ai/Yi/discussions" target="_blank"> GitHub </a>
</p> 

<p align="center">
    👋 Join us on <a href="https://discord.gg/hYUwWddeAu" target="_blank"> 👾 Discord </a> or <a href="https://github.com/01-ai/Yi/issues/43#issuecomment-1827285245" target="_blank"> 💬 WeChat </a>
</p> 

<p align="center">
    📝 Check out  <a href="https://arxiv.org/abs/2403.04652"> Yi Tech Report </a>
</p> 

<p align="center">
    📚 Grow at <a href="https://github.com/catchygit/Yi/wiki"> Yi Wiki </a>
</p> 

<!-- DO NOT REMOVE ME -->

<hr>

🤖 The Yi series models are the next generation of open-source large language models trained from scratch by [01.AI](https://01.ai/). Targeted as a bilingual language model and trained on 3T multilingual corpus, the Yi series models become one of the strongest LLM worldwide, showing promise in language understanding, commonsense reasoning, reading comprehension, and more.

<details open>
<summary></b>📕 Table of Contents</b></summary>

- [Models](#models)
- [Quick start](#quick-start)
- [Fine-tuning](#fine-tuning)
- [Quantization](#quantization)
- [Requirements](#requirements)
- [Misc.](#misc)

</details>

<hr>

## Models

Yi models come in multiple sizes and cater to different use cases. You can also fine-tune Yi models to meet your specific requirements. 

If you want to deploy Yi models, make sure you meet the [software and hardware requirements](#deployment).

### Chat models

| Model | Download  
|---|---
Yi-34B-Chat	| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-34B-Chat)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-34B-Chat/summary)
Yi-34B-Chat-4bits	| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-34B-Chat-4bits)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-34B-Chat-4bits/summary)
Yi-34B-Chat-8bits | • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-34B-Chat-8bits) • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-34B-Chat-8bits/summary)
Yi-6B-Chat| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-6B-Chat) • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-6B-Chat/summary)
Yi-6B-Chat-4bits |	• [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-6B-Chat-4bits)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-6B-Chat-4bits/summary)
Yi-6B-Chat-8bits	|  • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-6B-Chat-8bits) • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-6B-Chat-8bits/summary)


<sub><sup> - 4-bit series models are quantized by AWQ. <br> - 8-bit series models are quantized by GPTQ <br> - All quantized models have a low barrier to use since they can be deployed on consumer-grade GPUs (e.g., 3090, 4090). </sup></sub>

### Base models

| Model | Download | 
|---|---|
Yi-34B| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-34B)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-34B/summary)
Yi-34B-200K|• [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-34B-200K)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-34B-200K/summary)
Yi-9B|• [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-9B) • [🤖 ModelScope](https://wisemodel.cn/models/01.AI/Yi-9B)
Yi-9B-200K | • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-9B-200K)   • [🤖 ModelScope](https://wisemodel.cn/models/01.AI/Yi-9B-200K)
Yi-6B| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-6B)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-6B/summary)
Yi-6B-200K	| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-6B-200K) • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-6B-200K/summary)

<sub><sup> - 200k is roughly equivalent to 400,000 Chinese characters.  <br> - If you want to use the previous version of the Yi-34B-200K (released on Nov 5, 2023), run `git checkout 069cd341d60f4ce4b07ec394e82b79e94f656cf` to download the weight. </sup></sub>

### Model info

- For chat and base models

<table>
<thead>
<tr>
<th>Model</th>
<th>Intro</th>
<th>Default context window</th>
<th>Pretrained tokens</th>
<th>Training Data Date</th>
</tr>
</thead>
<tbody><tr>
<td>6B series models</td>
<td>They are suitable for personal and academic use.</td>
<td rowspan="3">4K</td>
<td>3T</td>
<td rowspan="3">Up to June 2023</td>
</tr>
<tr>
<td>9B series models</td>
<td>It is the best at coding and math in the Yi series models.</td>
<td>Yi-9B is continuously trained based on Yi-6B, using 0.8T tokens.</td>
</tr>
<tr>
<td>34B series models</td>
<td>They are suitable for personal, academic, and commercial (particularly for small and medium-sized enterprises) purposes. It&#39;s a cost-effective solution that&#39;s affordable and equipped with emergent ability.</td>
<td>3T</td>
</tr>
</tbody></table>


- For chat models
  
  <details style="display: inline;"><summary>For chat model limitations, see the explanations below. ⬇️</summary>
   <ul>
    <br>The released chat model has undergone exclusive training using Supervised Fine-Tuning (SFT). Compared to other standard chat models, our model produces more diverse responses, making it suitable for various downstream tasks, such as creative scenarios. Furthermore, this diversity is expected to enhance the likelihood of generating higher quality responses, which will be advantageous for subsequent Reinforcement Learning (RL) training.

    <br>However, this higher diversity might amplify certain existing issues, including:
      <li>Hallucination: This refers to the model generating factually incorrect or nonsensical information. With the model's responses being more varied, there's a higher chance of hallucination that are not based on accurate data or logical reasoning.</li>
      <li>Non-determinism in re-generation: When attempting to regenerate or sample responses, inconsistencies in the outcomes may occur. The increased diversity can lead to varying results even under similar input conditions.</li>
      <li>Cumulative Error: This occurs when errors in the model's responses compound over time. As the model generates more diverse responses, the likelihood of small inaccuracies building up into larger errors increases, especially in complex tasks like extended reasoning, mathematical problem-solving, etc.</li>
      <li>To achieve more coherent and consistent responses, it is advisable to adjust generation configuration parameters such as temperature, top_p, or top_k. These adjustments can help in the balance between creativity and coherence in the model's outputs.</li>
</ul>
</details>

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>


# Quick start

Getting up and running with Yi models is simple with multiple choices available. 

### Choose your path

Select one of the following paths to begin your journey with Yi!

 ![Quick start - Choose your path](https://github.com/01-ai/Yi/blob/main/assets/img/quick_start_path.png?raw=true)

#### 🎯 Deploy Yi locally

If you prefer to deploy Yi models locally, 

  - 🙋‍♀️ and you have **sufficient** resources (for example, NVIDIA A800 80GB), you can choose one of the following methods:
    - [pip](#quick-start---pip)
    - [Docker](#quick-start---docker)
    - [conda-lock](#quick-start---conda-lock)

  - 🙋‍♀️ and you have **limited** resources (for example, a MacBook Pro), you can use [llama.cpp](#quick-start---llamacpp).

#### 🎯 Not to deploy Yi locally

If you prefer not to deploy Yi models locally, you can explore Yi's capabilities using any of the following options.

##### 🙋‍♀️ Run Yi with APIs

If you want to explore more features of Yi, you can adopt one of these methods:

- Yi APIs (Yi official)
  - [Early access has been granted](https://x.com/01AI_Yi/status/1735728934560600536?s=20) to some applicants. Stay tuned for the next round of access!

- [Yi APIs](https://replicate.com/01-ai/yi-34b-chat/api?tab=nodejs) (Replicate)

##### 🙋‍♀️ Run Yi in playground

If you want to chat with Yi with more customizable options (e.g., system prompt, temperature, repetition penalty, etc.), you can try one of the following options:
  
  - [Yi-34B-Chat-Playground](https://platform.lingyiwanwu.com/prompt/playground) (Yi official)
    - Access is available through a whitelist. Welcome to apply (fill out a form in [English](https://cn.mikecrm.com/l91ODJf) or [Chinese](https://cn.mikecrm.com/gnEZjiQ)).
  
  - [Yi-34B-Chat-Playground](https://replicate.com/01-ai/yi-34b-chat) (Replicate) 

##### 🙋‍♀️ Chat with Yi

 If you want to chat with Yi, you can use one of these online services, which offer a similar user experience:

- [Yi-34B-Chat](https://huggingface.co/spaces/01-ai/Yi-34B-Chat) (Yi official on Hugging Face)
  - No registration is required.

- [Yi-34B-Chat](https://platform.lingyiwanwu.com/) (Yi official beta)
  - Access is available through a whitelist. Welcome to apply (fill out a form in [English](https://cn.mikecrm.com/l91ODJf) or [Chinese](https://cn.mikecrm.com/gnEZjiQ)).

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

### Quick start - pip 

This tutorial guides you through every step of running **Yi-34B-Chat locally on an A800 (80G)** and then performing inference.

#### Step 0: Prerequisites
 
- Make sure Python 3.10 or a later version is installed.

- If you want to run other Yi models, see [software and hardware requirements](#deployment).

#### Step 1: Prepare your environment 

To set up the environment and install the required packages, execute the following command.

```bash
git clone https://github.com/01-ai/Yi.git
cd yi
pip install -r requirements.txt
```

#### Step 2: Download the Yi model

You can download the weights and tokenizer of Yi models from the following sources:

- [Hugging Face](https://huggingface.co/01-ai)
- [ModelScope](https://www.modelscope.cn/organization/01ai/)
- [WiseModel](https://wisemodel.cn/organization/01.AI)

#### Step 3: Perform inference

You can perform inference with Yi chat or base models as below.

##### Perform inference with Yi chat model

1. Create a file named  `quick_start.py` and copy the following content to it.

    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = '<your-model-path>'

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    # Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype='auto'
    ).eval()

    # Prompt content: "hi"
    messages = [
        {"role": "user", "content": "hi"}
    ]

    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
    output_ids = model.generate(input_ids.to('cuda'))
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

    # Model response: "Hello! How can I assist you today?"
    print(response)
    ```

2. Run `quick_start.py`.

    ```bash
    python quick_start.py
    ```

    Then you can see an output similar to the one below. 🥳

    ```bash
    Hello! How can I assist you today?
    ```

##### Perform inference with Yi base model

- Yi-34B

  The steps are similar to [pip - Perform inference with Yi chat model](#perform-inference-with-yi-chat-model).

  You can use the existing file [`text_generation.py`](https://github.com/01-ai/Yi/tree/main/demo).

  ```bash
  python demo/text_generation.py  --model <your-model-path>
  ```

  Then you can see an output similar to the one below. 🥳

  <details>

  <summary>Output. ⬇️ </summary>

  <br>

  **Prompt**: Let me tell you an interesting story about cat Tom and mouse Jerry,

  **Generation**: Let me tell you an interesting story about cat Tom and mouse Jerry, which happened in my childhood. My father had a big house with two cats living inside it to kill mice. One day when I was playing at home alone, I found one of the tomcats lying on his back near our kitchen door, looking very much like he wanted something from us but couldn’t get up because there were too many people around him! He kept trying for several minutes before finally giving up...

  </details>

- Yi-9B
  
  Input

  ```bash
  from transformers import AutoModelForCausalLM, AutoTokenizer

  MODEL_DIR = "01-ai/Yi-9B"
  model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="auto")
  tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)

  input_text = "# write the quick sort algorithm"
  inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
  outputs = model.generate(**inputs, max_length=256)
  print(tokenizer.decode(outputs[0], skip_special_tokens=True))
  ```

  Output

  ```bash
  # write the quick sort algorithm
  def quick_sort(arr):
      if len(arr) <= 1:
          return arr
      pivot = arr[len(arr) // 2]
      left = [x for x in arr if x < pivot]
      middle = [x for x in arr if x == pivot]
      right = [x for x in arr if x > pivot]
      return quick_sort(left) + middle + quick_sort(right)

  # test the quick sort algorithm
  print(quick_sort([3, 6, 8, 10, 1, 2, 1]))
  ```


<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

### Quick start - Docker 
<details>
<summary> Run Yi-34B-chat locally with Docker: a step-by-step guide. ⬇️</summary> 
<br>This tutorial guides you through every step of running <strong>Yi-34B-Chat on an A800 GPU</strong> or <strong>4*4090</strong> locally and then performing inference.
 <h4>Step 0: Prerequisites</h4>
<p>Make sure you've installed <a href="https://docs.docker.com/engine/install/?open_in_browser=true">Docker</a> and <a href="https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html">nvidia-container-toolkit</a>.</p>

<h4> Step 1: Start Docker </h4>
<pre><code>docker run -it --gpus all \
-v &lt;your-model-path&gt;: /models
ghcr.io/01-ai/yi:latest
</code></pre>
<p>Alternatively, you can pull the Yi Docker image from <code>registry.lingyiwanwu.com/ci/01-ai/yi:latest</code>.</p>

<h4>Step 2: Perform inference</h4>
    <p>You can perform inference with Yi chat or base models as below.</p>
    
<h5>Perform inference with Yi chat model</h5>
    <p>The steps are similar to <a href="#perform-inference-with-yi-chat-model">pip - Perform inference with Yi chat model</a>.</p>
    <p><strong>Note</strong> that the only difference is to set <code>model_path = '&lt;your-model-mount-path&gt;'</code> instead of <code>model_path = '&lt;your-model-path&gt;'</code>.</p>
<h5>Perform inference with Yi base model</h5>
    <p>The steps are similar to <a href="#perform-inference-with-yi-base-model">pip - Perform inference with Yi base model</a>.</p>
    <p><strong>Note</strong> that the only difference is to set <code>--model &lt;your-model-mount-path&gt;'</code> instead of <code>model &lt;your-model-path&gt;</code>.</p>
</details>

### Quick start - conda-lock

<details>
<summary>You can use <code><a href="https://github.com/conda/conda-lock">conda-lock</a></code> to generate fully reproducible lock files for conda environments. ⬇️</summary>
<br>
You can refer to <a href="https://github.com/01-ai/Yi/blob/ebba23451d780f35e74a780987ad377553134f68/conda-lock.yml">conda-lock.yml</a>  for the exact versions of the dependencies. Additionally, you can utilize <code><a href="https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html">micromamba</a></code> for installing these dependencies.
<br>
To install the dependencies, follow these steps:

1. Install micromamba by following the instructions available <a href="https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html">here</a>.

2. Execute <code>micromamba install -y -n yi -f conda-lock.yml</code> to create a conda environment named <code>yi</code> and install the necessary dependencies.
</details>


### Quick start - llama.cpp
<a href="https://github.com/01-ai/Yi/blob/main/docs/README_llama.cpp.md">The following tutorial </a> will guide you through every step of running a quantized model (<a href="https://huggingface.co/XeIaso/yi-chat-6B-GGUF/tree/main">Yi-chat-6B-2bits</a>) locally and then performing inference.
<details>
<summary> Run Yi-chat-6B-2bits locally with llama.cpp: a step-by-step guide. ⬇️</summary> 
<br><a href="https://github.com/01-ai/Yi/blob/main/docs/README_llama.cpp.md">This tutorial</a> guides you through every step of running a quantized model (<a href="https://huggingface.co/XeIaso/yi-chat-6B-GGUF/tree/main">Yi-chat-6B-2bits</a>) locally and then performing inference.</p>

- [Step 0: Prerequisites](#step-0-prerequisites)
- [Step 1: Download llama.cpp](#step-1-download-llamacpp)
- [Step 2: Download Yi model](#step-2-download-yi-model)
- [Step 3: Perform inference](#step-3-perform-inference)

#### Step 0: Prerequisites 

- This tutorial assumes you use a MacBook Pro with 16GB of memory and an Apple M2 Pro chip.
  
- Make sure [`git-lfs`](https://git-lfs.com/) is installed on your machine.
  
#### Step 1: Download `llama.cpp`

To clone the [`llama.cpp`](https://github.com/ggerganov/llama.cpp) repository, run the following command.

```bash
git clone git@github.com:ggerganov/llama.cpp.git
```

#### Step 2: Download Yi model

2.1 To clone [XeIaso/yi-chat-6B-GGUF](https://huggingface.co/XeIaso/yi-chat-6B-GGUF/tree/main) with just pointers, run the following command.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/XeIaso/yi-chat-6B-GGUF
```

2.2 To download a quantized Yi model ([yi-chat-6b.Q2_K.gguf](https://huggingface.co/XeIaso/yi-chat-6B-GGUF/blob/main/yi-chat-6b.Q2_K.gguf)), run the following command.

```bash
git-lfs pull --include yi-chat-6b.Q2_K.gguf
```

#### Step 3: Perform inference

To perform inference with the Yi model, you can use one of the following methods.

- [Method 1: Perform inference in terminal](#method-1-perform-inference-in-terminal)
  
- [Method 2: Perform inference in web](#method-2-perform-inference-in-web)

##### Method 1: Perform inference in terminal

To compile `llama.cpp` using 4 threads and then conduct inference, navigate to the `llama.cpp` directory, and run the following command.

> ##### Tips
> 
> - Replace `/Users/yu/yi-chat-6B-GGUF/yi-chat-6b.Q2_K.gguf` with the actual path of your model.
>
> - By default, the model operates in completion mode.
> 
> - For additional output customization options (for example, system prompt, temperature, repetition penalty, etc.), run `./main -h` to check detailed descriptions and usage.

```bash
make -j4 && ./main -m /Users/yu/yi-chat-6B-GGUF/yi-chat-6b.Q2_K.gguf -p "How do you feed your pet fox? Please answer this question in 6 simple steps:\nStep 1:" -n 384 -e

...

How do you feed your pet fox? Please answer this question in 6 simple steps:

Step 1: Select the appropriate food for your pet fox. You should choose high-quality, balanced prey items that are suitable for their unique dietary needs. These could include live or frozen mice, rats, pigeons, or other small mammals, as well as fresh fruits and vegetables.

Step 2: Feed your pet fox once or twice a day, depending on the species and its individual preferences. Always ensure that they have access to fresh water throughout the day.

Step 3: Provide an appropriate environment for your pet fox. Ensure it has a comfortable place to rest, plenty of space to move around, and opportunities to play and exercise.

Step 4: Socialize your pet with other animals if possible. Interactions with other creatures can help them develop social skills and prevent boredom or stress.

Step 5: Regularly check for signs of illness or discomfort in your fox. Be prepared to provide veterinary care as needed, especially for common issues such as parasites, dental health problems, or infections.

Step 6: Educate yourself about the needs of your pet fox and be aware of any potential risks or concerns that could affect their well-being. Regularly consult with a veterinarian to ensure you are providing the best care.

...

```

Now you have successfully asked a question to the Yi model and got an answer! 🥳

##### Method 2: Perform inference in web

1. To initialize a lightweight and swift chatbot, run the following command.

    ```bash
    cd llama.cpp
    ./server --ctx-size 2048 --host 0.0.0.0 --n-gpu-layers 64 --model /Users/yu/yi-chat-6B-GGUF/yi-chat-6b.Q2_K.gguf
    ```

    Then you can get an output like this:


    ```bash
    ...

    llama_new_context_with_model: n_ctx      = 2048
    llama_new_context_with_model: freq_base  = 5000000.0
    llama_new_context_with_model: freq_scale = 1
    ggml_metal_init: allocating
    ggml_metal_init: found device: Apple M2 Pro
    ggml_metal_init: picking default device: Apple M2 Pro
    ggml_metal_init: ggml.metallib not found, loading from source
    ggml_metal_init: GGML_METAL_PATH_RESOURCES = nil
    ggml_metal_init: loading '/Users/yu/llama.cpp/ggml-metal.metal'
    ggml_metal_init: GPU name:   Apple M2 Pro
    ggml_metal_init: GPU family: MTLGPUFamilyApple8 (1008)
    ggml_metal_init: hasUnifiedMemory              = true
    ggml_metal_init: recommendedMaxWorkingSetSize  = 11453.25 MB
    ggml_metal_init: maxTransferRate               = built-in GPU
    ggml_backend_metal_buffer_type_alloc_buffer: allocated buffer, size =   128.00 MiB, ( 2629.44 / 10922.67)
    llama_new_context_with_model: KV self size  =  128.00 MiB, K (f16):   64.00 MiB, V (f16):   64.00 MiB
    ggml_backend_metal_buffer_type_alloc_buffer: allocated buffer, size =     0.02 MiB, ( 2629.45 / 10922.67)
    llama_build_graph: non-view tensors processed: 676/676
    llama_new_context_with_model: compute buffer total size = 159.19 MiB
    ggml_backend_metal_buffer_type_alloc_buffer: allocated buffer, size =   156.02 MiB, ( 2785.45 / 10922.67)
    Available slots:
    -> Slot 0 - max context: 2048

    llama server listening at http://0.0.0.0:8080
    ```

2. To access the chatbot interface, open your web browser and enter `http://0.0.0.0:8080` into the address bar. 
   
    ![Yi model chatbot interface - llama.cpp](https://github.com/01-ai/Yi/blob/main/assets/img/yi_llama_cpp1.png?raw=true)


3. Enter a question, such as "How do you feed your pet fox? Please answer this question in 6 simple steps" into the prompt window, and you will receive a corresponding answer.

    ![Ask a question to Yi model - llama.cpp](https://github.com/01-ai/Yi/blob/main/assets/img/yi_llama_cpp2.png?raw=true)

</ul>
</details>

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

### Web demo

You can build a web UI demo for Yi **chat** models (note that Yi base models are not supported in this senario).

[Step 1: Prepare your environment](#step-1-prepare-your-environment). 

[Step 2: Download the Yi model](#step-2-download-the-yi-model).

Step 3. To start a web service locally, run the following command.

```bash
python demo/web_demo.py -c <your-model-path>
```

You can access the web UI by entering the address provided in the console into your browser. 

 ![Quick start - web demo](https://github.com/01-ai/Yi/blob/main/assets/img/yi_34b_chat_web_demo.gif?raw=true)

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

### Fine-tuning

[For details on fine-tuning.](https://github.com/catchygit/Yi/wiki/Fine-tuning)

### Quantization

[For details on quantization.](https://github.com/catchygit/Yi/wiki/Quantization)

### Requirements

If you want to deploy Yi models, make sure you meet the software and hardware requirements. 

#### Software requirements

Before using Yi quantized models, make sure you've installed the correct software listed below.

| Model | Software
|---|---
Yi 4-bit quantized models | [AWQ and CUDA](https://github.com/casper-hansen/AutoAWQ?tab=readme-ov-file#install-from-pypi)
Yi 8-bit quantized models |  [GPTQ and CUDA](https://github.com/PanQiWei/AutoGPTQ?tab=readme-ov-file#quick-installation)

#### Hardware requirements

Before deploying Yi in your environment, make sure your hardware meets the following requirements.

##### Chat models

| Model                | Minimum VRAM |        Recommended GPU Example       |
|:----------------------|:--------------|:-------------------------------------:|
| Yi-6B-Chat           | 15 GB         | 1 x RTX 3090 (24 GB) <br> 1 x RTX 4090 (24 GB) <br>  1 x A10 (24 GB)  <br> 1 x A30 (24 GB)              |
| Yi-6B-Chat-4bits     | 4 GB          | 1 x RTX 3060 (12 GB)<br> 1 x RTX 4060 (8 GB)                   |
| Yi-6B-Chat-8bits     | 8 GB          | 1 x RTX 3070 (8 GB) <br> 1 x RTX 4060 (8 GB)                   |
| Yi-34B-Chat          | 72 GB         | 4 x RTX 4090 (24 GB)<br> 1 x A800 (80GB)               |
| Yi-34B-Chat-4bits    | 20 GB         | 1 x RTX 3090 (24 GB) <br> 1 x RTX 4090 (24 GB) <br> 1 x A10 (24 GB)  <br> 1 x A30 (24 GB)  <br> 1 x A100 (40 GB) |
| Yi-34B-Chat-8bits    | 38 GB         | 2 x RTX 3090 (24 GB) <br> 2 x RTX 4090 (24 GB)<br> 1 x A800  (40 GB) |

Below are detailed minimum VRAM requirements under different batch use cases.

|  Model                  | batch=1 | batch=4 | batch=16 | batch=32 |
| ----------------------- | ------- | ------- | -------- | -------- |
| Yi-6B-Chat              | 12 GB   | 13 GB   | 15 GB    | 18 GB    |
| Yi-6B-Chat-4bits  | 4 GB    | 5 GB    | 7 GB     | 10 GB    |
| Yi-6B-Chat-8bits  | 7 GB    | 8 GB    | 10 GB    | 14 GB    |
| Yi-34B-Chat       | 65 GB   | 68 GB   | 76 GB    | > 80 GB   |
| Yi-34B-Chat-4bits | 19 GB   | 20 GB   | 30 GB    | 40 GB    |
| Yi-34B-Chat-8bits | 35 GB   | 37 GB   | 46 GB    | 58 GB    |

##### Base models

| Model                | Minimum VRAM |        Recommended GPU Example       |
|----------------------|--------------|:-------------------------------------:|
| Yi-6B                | 15 GB         | 1 x RTX 3090 (24 GB) <br> 1 x RTX 4090 (24 GB) <br> 1 x A10 (24 GB)  <br> 1 x A30 (24 GB)                |
| Yi-6B-200K           | 50 GB         | 1 x A800 (80 GB)                            |
| Yi-9B                | 20 GB         | 1 x RTX 4090 (24 GB)                           |
| Yi-34B               | 72 GB         | 4 x RTX 4090 (24 GB) <br> 1 x A800 (80 GB)               |
| Yi-34B-200K          | 200 GB        | 4 x A800 (80 GB)                        |

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

## Benchmarks 

See the following pages for detailed benchmarking:
- [Chat model performance](https://github.com/catchygit/Yi/wiki/Benchmarks#chat-model-performance)
- [Base model performance](https://github.com/catchygit/Yi/wiki/Benchmarks#base-model-performance)

# Misc.

### Who can use Yi?

Everyone! 🙌 ✅

- The Yi series models are free for personal usage, academic purposes, and commercial use. All usage must adhere to the [Yi Series Models Community License Agreement 2.1](https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt)
  
- For free commercial use, you only need to [complete this form](https://www.lingyiwanwu.com/yi-license) to get a Yi Model Commercial License.

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

### Acknowledgments

A heartfelt thank you to each of you who have made contributions to the Yi community! You have helped Yi not just a project, but a vibrant, growing home for innovation.

[![yi contributors](https://contrib.rocks/image?repo=01-ai/yi&max=2000&columns=15)](https://github.com/01-ai/yi/graphs/contributors)

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

### Disclaimer

We use data compliance checking algorithms during the training process, to
ensure the compliance of the trained model to the best of our ability. Due to
complex data and the diversity of language model usage scenarios, we cannot
guarantee that the model will generate correct, and reasonable output in all
scenarios. Please be aware that there is still a risk of the model producing
problematic outputs. We will not be responsible for any risks and issues
resulting from misuse, misguidance, illegal usage, and related misinformation,
as well as any associated data security concerns.

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

### License

The source code in this repo is licensed under the [Apache 2.0
license](https://github.com/01-ai/Yi/blob/main/LICENSE). The Yi series models are fully open for academic research and free for commercial use, with automatic permission granted upon application. All usage must adhere to the [Yi Series Models Community License Agreement 2.1](https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt).
For free commercial use, you only need to send an email to [get official commercial permission](https://www.lingyiwanwu.com/yi-license).

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>
