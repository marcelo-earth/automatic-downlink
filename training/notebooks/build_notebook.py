#!/usr/bin/env python3
"""Build the Kaggle notebook JSON for save_notebook API."""
import json

cells = []

# Cell 0: Markdown
cells.append({
    "cell_type": "markdown",
    "source": (
        "# Fine-tune LFM2.5-VL-450M for Satellite Triage\n\n"
        "**Kaggle GPU runtime required** (T4). "
        "This notebook fine-tunes the Liquid AI VLM on satellite imagery to produce triage JSON.\n\n"
        "All cells run top-to-bottom. Merge + test + upload happen automatically after training."
    ),
    "metadata": {}
})

# Cell 1: Install
cells.append({
    "cell_type": "code",
    "source": "!pip install -q transformers peft accelerate bitsandbytes datasets huggingface_hub pillow",
    "metadata": {"trusted": True}, "outputs": [], "execution_count": None
})

# Cell 2: HF login
cells.append({
    "cell_type": "code",
    "source": (
        "import os\n"
        "from huggingface_hub import login, hf_hub_download\n\n"
        "from kaggle_secrets import UserSecretsClient\n"
        "secrets = UserSecretsClient()\n"
        "HF_TOKEN = secrets.get_secret(\"HF_TOKEN\")\n"
        "login(token=HF_TOKEN)"
    ),
    "metadata": {"trusted": True}, "outputs": [], "execution_count": None
})

# Cell 3: Download labels
cells.append({
    "cell_type": "code",
    "source": (
        "from huggingface_hub import snapshot_download\n\n"
        "snapshot_download(\n"
        "    \"marcelo-earth/VRSBench-satellite-triage-labels\",\n"
        "    repo_type=\"dataset\",\n"
        "    local_dir=\"/kaggle/working/labels\",\n"
        ")\n"
        "!ls /kaggle/working/labels/"
    ),
    "metadata": {"trusted": True}, "outputs": [], "execution_count": None
})

# Cell 4: Download images
cells.append({
    "cell_type": "code",
    "source": (
        "import zipfile\n\n"
        "zip_path = hf_hub_download(\"xiang709/VRSBench\", \"Images_train.zip\", repo_type=\"dataset\")\n"
        "with zipfile.ZipFile(zip_path) as zf:\n"
        "    zf.extractall(\"/kaggle/working/images\")\n"
        "!ls /kaggle/working/images/ | head -5\n"
        "!ls /kaggle/working/images/Images_train/ | wc -l"
    ),
    "metadata": {"trusted": True}, "outputs": [], "execution_count": None
})

SYSTEM_PROMPT = (
    "You are a satellite image triage system. Analyze the image and respond ONLY with a JSON object. No other text.\\n\\n"
    "Priority: CRITICAL (disasters, fires, floods), HIGH (deforestation, unusual activity, anomalies), "
    "MEDIUM (routine urban, agriculture), LOW (featureless desert, barren terrain), "
    "SKIP (heavy clouds >80%, empty ocean, image artifacts).\\n\\n"
    "If the image is mostly white/bright with no ground features visible, it is cloud-covered — mark SKIP.\\n\\n"
    "Examples:\\n"
    '{"description": "Dense urban area with buildings and road network along a coastline", "priority": "MEDIUM", "reasoning": "Routine urban scene, no anomalies detected", "categories": ["urban", "infrastructure"]}\\n'
    '{"description": "Image almost entirely covered by clouds, no ground features visible", "priority": "SKIP", "reasoning": "Cloud cover exceeds 80%, no usable data", "categories": ["cloud_cover"]}\\n'
    '{"description": "Arid desert terrain with sand dunes and dry riverbeds", "priority": "LOW", "reasoning": "Featureless barren landscape with no activity", "categories": ["terrain", "desert"]}\\n'
    '{"description": "Active wildfire with visible smoke plumes spreading over forested area", "priority": "CRITICAL", "reasoning": "Active fire threatening forested region, immediate alert needed", "categories": ["disaster", "fire", "vegetation"]}\\n'
    '{"description": "Fresh clearing in dense forest with exposed soil and new access road", "priority": "HIGH", "reasoning": "Possible deforestation activity with new road construction", "categories": ["deforestation", "vegetation", "environmental_change"]}'
)

# Cell 5: Build dataset — all samples, no downsampling
cells.append({
    "cell_type": "code",
    "source": (
        "import json, random\n"
        "from collections import defaultdict, Counter\n\n"
        "with open(\"/kaggle/working/labels/exp5_captions.jsonl\") as f:\n"
        "    captions = [json.loads(l) for l in f]\n"
        "with open(\"/kaggle/working/labels/exp5_labels.jsonl\") as f:\n"
        "    labels = [json.loads(l) for l in f]\n\n"
        "assert len(captions) == len(labels)\n"
        "print(f\"Total samples: {len(captions)}\")\n\n"
        f"SYSTEM_PROMPT = \"{SYSTEM_PROMPT}\"\n\n"
        "USER_PROMPT = \"Triage this satellite image. Respond with JSON only.\"\n\n"
        "all_samples = []\n"
        "img_dir = \"/kaggle/working/images/Images_train\"\n\n"
        "for cap, lab in zip(captions, labels):\n"
        "    img_path = f\"{img_dir}/{cap['image']}\"\n"
        "    if not os.path.exists(img_path):\n"
        "        continue\n"
        "    triage_json = json.dumps({\n"
        "        \"description\": cap[\"caption\"],\n"
        "        \"priority\": lab[\"priority\"],\n"
        "        \"reasoning\": lab[\"reasoning\"],\n"
        "        \"categories\": lab[\"categories\"],\n"
        "    })\n"
        "    sample = {\n"
        "        \"messages\": [\n"
        "            {\"role\": \"system\", \"content\": [{\"type\": \"text\", \"text\": SYSTEM_PROMPT}]},\n"
        "            {\"role\": \"user\", \"content\": [\n"
        "                {\"type\": \"image\", \"image\": img_path},\n"
        "                {\"type\": \"text\", \"text\": USER_PROMPT},\n"
        "            ]},\n"
        "            {\"role\": \"assistant\", \"content\": [{\"type\": \"text\", \"text\": triage_json}]},\n"
        "        ],\n"
        "        \"priority\": lab[\"priority\"],\n"
        "    }\n"
        "    all_samples.append(sample)\n\n"
        "dist = Counter(s['priority'] for s in all_samples)\n"
        "for p in ['CRITICAL','HIGH','MEDIUM','LOW','SKIP']:\n"
        "    print(f\"  {p}: {dist.get(p,0)}\")\n"
        "print(f\"Total: {len(all_samples)}\")"
    ),
    "metadata": {"trusted": True}, "outputs": [], "execution_count": None
})

# Cell 6: Stratified 90/10 split — all samples, no downsampling
cells.append({
    "cell_type": "code",
    "source": (
        "random.seed(42)\n\n"
        "by_priority = defaultdict(list)\n"
        "for s in all_samples:\n"
        "    by_priority[s['priority']].append(s)\n\n"
        "train_data, eval_data = [], []\n"
        "for priority, samples in by_priority.items():\n"
        "    random.shuffle(samples)\n"
        "    split = max(1, int(len(samples) * 0.9))\n"
        "    train_data.extend(samples[:split])\n"
        "    eval_data.extend(samples[split:])\n\n"
        "random.shuffle(train_data)\n"
        "random.shuffle(eval_data)\n\n"
        "print(f\"Train: {len(train_data)}, Eval: {len(eval_data)}\")\n\n"
        "with open(\"/kaggle/working/train.jsonl\", \"w\") as f:\n"
        "    for s in train_data:\n"
        "        f.write(json.dumps(s) + \"\\n\")\n"
        "with open(\"/kaggle/working/eval.jsonl\", \"w\") as f:\n"
        "    for s in eval_data:\n"
        "        f.write(json.dumps(s) + \"\\n\")\n"
        "print('Train priorities:', Counter(s['priority'] for s in train_data))\n"
        "print('Eval priorities:', Counter(s['priority'] for s in eval_data))"
    ),
    "metadata": {"trusted": True}, "outputs": [], "execution_count": None
})

# Cell 7: Load model + LoRA
cells.append({
    "cell_type": "code",
    "source": (
        "import torch\n"
        "from transformers import AutoModelForImageTextToText, AutoProcessor\n"
        "from peft import LoraConfig, get_peft_model\n\n"
        "BASE_MODEL = \"LiquidAI/LFM2.5-VL-450M\"\n\n"
        "print(\"Loading model...\")\n"
        "model = AutoModelForImageTextToText.from_pretrained(\n"
        "    BASE_MODEL, dtype=torch.bfloat16, device_map=\"auto\"\n"
        ")\n"
        "processor = AutoProcessor.from_pretrained(BASE_MODEL)\n\n"
        "lora_config = LoraConfig(\n"
        "    r=8,\n"
        "    lora_alpha=16,\n"
        "    lora_dropout=0.1,\n"
        "    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"out_proj\", \"fc1\", \"fc2\", \"in_proj\", \"linear_1\", \"linear_2\"],\n"
        "    task_type=\"CAUSAL_LM\",\n"
        ")\n\n"
        "model = get_peft_model(model, lora_config)\n"
        "model.print_trainable_parameters()"
    ),
    "metadata": {"trusted": True}, "outputs": [], "execution_count": None
})

# Cell 8: Dataset class
cells.append({
    "cell_type": "code",
    "source": (
        "import torch\n"
        "from torch.utils.data import Dataset\n"
        "from PIL import Image\n\n"
        "CLASS_WEIGHTS = {\n"
        "    'CRITICAL': 50.0, 'HIGH': 5.0, 'MEDIUM': 1.0, 'LOW': 2.0, 'SKIP': 3.0,\n"
        "}\n\n\n"
        "class TriageDataset(Dataset):\n"
        "    def __init__(self, jsonl_path, processor):\n"
        "        with open(jsonl_path) as f:\n"
        "            self.samples = [json.loads(l) for l in f]\n"
        "        self.processor = processor\n\n"
        "    def __len__(self):\n"
        "        return len(self.samples)\n\n"
        "    def __getitem__(self, idx):\n"
        "        sample = self.samples[idx]\n"
        "        messages = sample['messages']\n"
        "        priority = sample.get('priority', 'MEDIUM')\n\n"
        "        img_path = None\n"
        "        for msg in messages:\n"
        "            for c in msg['content']:\n"
        "                if isinstance(c, dict) and c.get('type') == 'image':\n"
        "                    img_path = c['image']\n"
        "        image = Image.open(img_path).convert('RGB')\n\n"
        "        proc_messages = []\n"
        "        for msg in messages:\n"
        "            new_content = []\n"
        "            for c in msg['content']:\n"
        "                if isinstance(c, dict) and c.get('type') == 'image':\n"
        "                    new_content.append({'type': 'image', 'image': image})\n"
        "                else:\n"
        "                    new_content.append(c)\n"
        "            proc_messages.append({'role': msg['role'], 'content': new_content})\n\n"
        "        inputs = self.processor.apply_chat_template(\n"
        "            proc_messages,\n"
        "            add_generation_prompt=False,\n"
        "            return_tensors='pt',\n"
        "            return_dict=True,\n"
        "            tokenize=True,\n"
        "        )\n"
        "        input_ids = inputs['input_ids'].squeeze(0)\n"
        "        inputs['labels'] = input_ids.clone()\n"
        "        inputs['class_weight'] = torch.tensor(CLASS_WEIGHTS.get(priority, 1.0))\n"
        "        return {k: v.squeeze(0) if v.dim() > 1 else v for k, v in inputs.items()}\n\n\n"
        "train_dataset = TriageDataset('/kaggle/working/train.jsonl', processor)\n"
        "eval_dataset = TriageDataset('/kaggle/working/eval.jsonl', processor)\n"
        "print(f\"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}\")\n"
        "sample = train_dataset[0]\n"
        "print(f\"Keys: {list(sample.keys())}\")\n"
        "print(f\"input_ids shape: {sample['input_ids'].shape}\")"
    ),
    "metadata": {"trusted": True}, "outputs": [], "execution_count": None
})

# Cell 9: Training with sample-level class-weighted loss
cells.append({
    "cell_type": "code",
    "source": (
        "from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq\n"
        "import torch.nn.functional as F\n\n\n"
        "class VLMTrainer(Trainer):\n"
        "    \"\"\"Custom trainer with sample-level class-weighted loss.\"\"\"\n"
        "    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):\n"
        "        labels = inputs.pop('labels')\n"
        "        weights = inputs.pop('class_weight')  # shape: (batch,)\n"
        "        outputs = model(**inputs)\n"
        "        logits = outputs.logits\n"
        "        shift_logits = logits[..., :-1, :].contiguous()\n"
        "        shift_labels = labels[..., 1:].contiguous()\n\n"
        "        batch_size = shift_logits.size(0)\n"
        "        total_loss = 0.0\n"
        "        for i in range(batch_size):\n"
        "            sample_loss = F.cross_entropy(\n"
        "                shift_logits[i].view(-1, shift_logits.size(-1)),\n"
        "                shift_labels[i].view(-1),\n"
        "                ignore_index=-100,\n"
        "            )\n"
        "            total_loss += sample_loss * weights[i]\n\n"
        "        loss = total_loss / batch_size\n"
        "        return (loss, outputs) if return_outputs else loss\n\n\n"
        "training_args = TrainingArguments(\n"
        "    output_dir='/kaggle/working/checkpoints',\n"
        "    num_train_epochs=2,\n"
        "    per_device_train_batch_size=1,\n"
        "    gradient_accumulation_steps=4,\n"
        "    per_device_eval_batch_size=1,\n"
        "    learning_rate=1e-4,\n"
        "    warmup_steps=50,\n"
        "    lr_scheduler_type='cosine',\n"
        "    eval_strategy='epoch',\n"
        "    save_strategy='epoch',\n"
        "    logging_steps=20,\n"
        "    bf16=True,\n"
        "    gradient_checkpointing=True,\n"
        "    dataloader_pin_memory=False,\n"
        "    remove_unused_columns=False,\n"
        "    report_to='none',\n"
        ")\n\n"
        "trainer = VLMTrainer(\n"
        "    model=model,\n"
        "    args=training_args,\n"
        "    train_dataset=train_dataset,\n"
        "    eval_dataset=eval_dataset,\n"
        "    data_collator=DataCollatorForSeq2Seq(processor.tokenizer, padding=True),\n"
        ")\n\n"
        "trainer.train()"
    ),
    "metadata": {"trusted": True}, "outputs": [], "execution_count": None
})

# Cell 10: COMBINED merge + test + upload
cells.append({
    "cell_type": "code",
    "source": (
        "# === MERGE + TEST + UPLOAD — all in one cell ===\n"
        "import subprocess\n\n"
        "print(\"=\" * 60)\n"
        "print(\"STEP 1: Merging LoRA adapter with base model...\")\n"
        "print(\"=\" * 60)\n\n"
        "merged = model.merge_and_unload()\n"
        "merged.save_pretrained(\"/kaggle/working/merged_model\")\n"
        "processor.save_pretrained(\"/kaggle/working/merged_model\")\n"
        "print(\"Merged model saved!\")\n\n"
        "print()\n"
        "print(\"=\" * 60)\n"
        "print(\"STEP 2: Quick inference test on 4 random images...\")\n"
        "print(\"=\" * 60)\n\n"
        "from transformers import AutoModelForImageTextToText\n\n"
        "test_model = AutoModelForImageTextToText.from_pretrained(\n"
        "    \"/kaggle/working/merged_model\", dtype=torch.bfloat16, device_map=\"auto\"\n"
        ")\n"
        "test_model.eval()\n\n"
        "result = subprocess.run(\n"
        "    \"ls /kaggle/working/images/Images_train/ | shuf | head -4\",\n"
        "    shell=True, capture_output=True, text=True\n"
        ")\n"
        "test_images = result.stdout.strip().split(\"\\n\")\n\n"
        "for img_name in test_images:\n"
        "    img = Image.open(f\"/kaggle/working/images/Images_train/{img_name}\").convert(\"RGB\")\n"
        "    conv = [\n"
        "        {\"role\": \"system\", \"content\": [{\"type\": \"text\", \"text\": SYSTEM_PROMPT}]},\n"
        "        {\"role\": \"user\", \"content\": [{\"type\": \"image\", \"image\": img}, {\"type\": \"text\", \"text\": USER_PROMPT}]},\n"
        "    ]\n"
        "    inputs = processor.apply_chat_template(\n"
        "        conv, add_generation_prompt=True, return_tensors=\"pt\", return_dict=True, tokenize=True\n"
        "    ).to(test_model.device)\n"
        "    with torch.inference_mode():\n"
        "        out = test_model.generate(**inputs, max_new_tokens=256, do_sample=False)\n"
        "    response = processor.decode(out[0, inputs[\"input_ids\"].shape[1]:], skip_special_tokens=True)\n"
        "    print(f\"\\n{img_name}: {response[:300]}\")\n\n"
        "del test_model\n"
        "torch.cuda.empty_cache()\n\n"
        "print()\n"
        "print(\"=\" * 60)\n"
        "print(\"STEP 3: Uploading to HuggingFace...\")\n"
        "print(\"=\" * 60)\n\n"
        "from huggingface_hub import HfApi\n\n"
        "api = HfApi()\n"
        "api.upload_folder(\n"
        "    folder_path=\"/kaggle/working/merged_model\",\n"
        "    repo_id=\"marcelo-earth/LFM2.5-VL-450M-satellite-triage\",\n"
        "    repo_type=\"model\",\n"
        "    commit_message=\"Exp 5: 20K samples, aligned prompt, clean captions, class-weighted loss (Kaggle T4)\",\n"
        ")\n\n"
        "print()\n"
        "print(\"=\" * 60)\n"
        "print(\"DONE! Model uploaded to marcelo-earth/LFM2.5-VL-450M-satellite-triage\")\n"
        "print(\"=\" * 60)"
    ),
    "metadata": {"trusted": True}, "outputs": [], "execution_count": None
})

notebook = {
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "name": "python", "version": "3.12.12", "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3", "nbconvert_exporter": "python", "file_extension": ".py"
        },
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "dataSources": [],
            "dockerImageVersionId": 31329,
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": True
        }
    },
    "nbformat_minor": 4,
    "nbformat": 4,
    "cells": cells
}

nb_json = json.dumps(notebook)
print(f"Cells: {len(cells)}")
print(f"JSON length: {len(nb_json)}")

with open("/Users/marcelo/Documents/GitHub/automatic-downlink/training/notebooks/notebook_payload.json", "w") as f:
    f.write(nb_json)
print("Saved to notebook_payload.json")
