# 🖲️ WEPO: Web Element Preference Optimization for LLM-based Web Navigation

Welcome to the official repository of **WEPO**, a novel approach for optimizing web element preferences in Large Language Model (LLM)-based web navigation tasks.

This repository contains the implementation of **WEPO** (Web Element Preference Optimization), which leverages **unsupervised preference learning** for contrastive training using **distance-based negative sampling**. Our method optimizes the interaction of LLMs with web elements by selecting relevant elements that align with user intent, improving web navigation and interaction efficiency.

The paper detailing this approach has been **accepted at AAAI 2025**. You can find more details in the full paper.

> **[Hugging Face Model: WEPO-Llama-3-8b](https://huggingface.co/KLGR123/WEPO-llama-3-8b)**  
> (Pre-trained model for the WEPO framework)

> **[Arxiv Link](https://arxiv.org/pdf/2412.10742)**

---

### Table of Contents

- [WEPO Implementation](#wepo-implementation)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Cite Us](#cite-us)
- [License](#license)

---

### 📑 WEPO Implementation

**WEPO** leverages **distance-based sampling** and **Direct Preference Optimization (DPO)** for optimizing LLM-based web navigation tasks. The implementation focuses on the DOM (Document Object Model) structure, sampling relevant and non-relevant web elements based on their distance within the DOM tree.

The main steps of the **WEPO** implementation include:

1. **DOM Parsing & Pruning**: The HTML of a web page is parsed into a DOM tree. A pruning mechanism isolates the most relevant elements for training by traversing the DOM and focusing on key elements and their ancestors.
   
2. **Distance-based Negative Sampling**: The method samples non-salient (negative) web elements based on their distance from relevant (positive) elements in the DOM. These samples are used in a contrastive learning setup.

3. **Direct Preference Optimization (DPO)**: We fine-tune a pretrained model using DPO, which optimizes the likelihood of operations on preferred elements and minimizes the likelihood of dis-preferred elements. The method avoids reward modeling and ensures stable training.

4. **Negative Sampling and Operation Diversity**: During training, a heuristic rule is applied to balance the sample types and ensure diverse operation types are used (e.g., `CLICK`, `TYPE`, `SELECT`).

---

### 🔩 Installation

To install WEPO, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/KLGR123/WEPO.git
   cd WEPO
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
---

### ⚙️ Usage

Below is the reference code for inference. First load the tokenizer and the model.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("KLGR123/WEPO-llama-3-8b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("KLGR123/WEPO-llama-3-8b", trust_remote_code=True).to('cuda:0')
```

Run a test-demo with random input.

```python
messages = [
    {"role": "system", "content": "You are a web navigation intelligence who interacts with webpage environments to achieve human user intent."},
    {"role": "user", "content": "Who are you?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=128,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.2,
    top_p=0.9,
)

response = outputs[0][input_ids.shape[-1]:]
output = tokenizer.decode(response, skip_special_tokens=True)
output
```

---

### 🎯 Cite Us

If you use **WEPO** in your research or applications, please cite our work:

```
@article{wepo2024,
  title={WEPO: Web Element Preference Optimization for LLM-based Web Navigation},
  author={Jiarun Liu, Jia Hao, Chunhong Zhang, Zheng Hu},
  journal={Arxiv},
  year={2024},
  url={MY_ARXIV_LINK}
}
```

For any inquiries, please reach out to us at [liujiarun01@bupt.edu.cn].

---

### 📍 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
