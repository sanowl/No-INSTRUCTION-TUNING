import os, sys, random, json, logging
from typing import List, Dict, Any
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoModelForCausalLM, AutoTokenizer, AdamW, get_cosine_schedule_with_warmup)
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

BASE_MODEL_NAME = "gpt2-large"
TOKENIZER_NAME = "gpt2-large"
INSTRUCTIONS_FILE = "instructions.jsonl"
RESPONSES_FILE = "responses.jsonl"
MODEL_SAVE_PATH = "fine-tuned-model"
NUM_EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 3e-5
MAX_GRAD_NORM = 1.0
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
MAX_LENGTH = 1024
SEED = 42

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]
    return data

class ResponseDataset(Dataset):
    def __init__(self, responses: List[str], tokenizer: AutoTokenizer, max_length: int = 1024):
        self.encodings = tokenizer(responses, truncation=True, max_length=max_length, padding='max_length', return_tensors='pt')
    def __len__(self):
        return self.encodings.input_ids.size(0)
    def __getitem__(self, idx):
        input_ids = self.encodings.input_ids[idx]
        attention_mask = self.encodings.attention_mask[idx]
        labels = input_ids.clone()
        labels[labels == self.encodings.tokenizer.pad_token_id] = -100
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

class InstructionResponseDataset(Dataset):
    def __init__(self, data: List[Dict[str, str]], tokenizer: AutoTokenizer, max_length: int = 1024):
        texts = [f"Instruction: {d['instruction']}\nResponse: {d['response']}" for d in data]
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length, padding='max_length', return_tensors='pt')
    def __len__(self):
        return self.encodings.input_ids.size(0)
    def __getitem__(self, idx):
        input_ids = self.encodings.input_ids[idx]
        attention_mask = self.encodings.attention_mask[idx]
        labels = input_ids.clone()
        labels[labels == self.encodings.tokenizer.pad_token_id] = -100
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

class RuleBasedAdapter:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.end_token_id = tokenizer.eos_token_id
        self.repetition_penalty = 1.2
        self.special_tokens_penalty = 0.5
        self.length_increment = 0.02
        self.word_penalties = {tokenizer.encode(w, add_special_tokens=False)[0]: 0.8 for w in ["the", "a", "an", "and", "or", "but", "because", "so", "if", "when", "then", "which", "who", "whom", "whose"]}
    def adjust_logits(self, logits: torch.Tensor, generated_tokens: List[int], step: int) -> torch.Tensor:
        logits[:, self.end_token_id] += step * self.length_increment
        for token_id in set(generated_tokens):
            logits[:, token_id] -= self.repetition_penalty
        for token_id in self.tokenizer.all_special_ids:
            logits[:, token_id] -= self.special_tokens_penalty
        for token_id, penalty in self.word_penalties.items():
            logits[:, token_id] *= penalty
        return logits

def train(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataset: Dataset, output_dir: str, num_epochs: int):
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    total_steps = len(data_loader) * num_epochs
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)
    scaler = torch.cuda.amp.GradScaler()
    model.train().to(device)
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / len(data_loader)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()
        logger.info(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def evaluate_response_ranking(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, data: List[Dict[str, Any]]) -> float:
    model.eval().to(device)
    correct = 0
    for item in tqdm(data, desc="Evaluating"):
        instruction = item['instruction']
        correct_response = item['correct_response']
        distractors = item['distractor_responses']
        responses = [correct_response] + distractors
        scores = []
        for response in responses:
            text = f"Instruction: {instruction}\nResponse: {response}"
            encoding = tokenizer(text, truncation=True, max_length=MAX_LENGTH, padding='max_length', return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(**encoding, labels=encoding['input_ids'])
                scores.append(-outputs.loss.item())
        if np.argmax(scores) == 0:
            correct += 1
    accuracy = correct / len(data)
    logger.info(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
    responses = load_jsonl(RESPONSES_FILE)
    instructions = load_jsonl(INSTRUCTIONS_FILE)
    response_texts = [entry['response'] for entry in responses]
    response_dataset = ResponseDataset(response_texts, tokenizer, MAX_LENGTH)
    train(base_model, tokenizer, response_dataset, os.path.join(MODEL_SAVE_PATH, "response_tuned"), NUM_EPOCHS)
    response_tuned_model = AutoModelForCausalLM.from_pretrained(os.path.join(MODEL_SAVE_PATH, "response_tuned"))
    poetry_data = [entry for entry in instructions if entry.get('task') == 'poetry']
    poetry_dataset = InstructionResponseDataset(poetry_data, tokenizer, MAX_LENGTH)
    train(base_model, tokenizer, poetry_dataset, os.path.join(MODEL_SAVE_PATH, "single_task_finetuned"), NUM_EPOCHS)
    single_task_model = AutoModelForCausalLM.from_pretrained(os.path.join(MODEL_SAVE_PATH, "single_task_finetuned"))
    adapter = RuleBasedAdapter(tokenizer)
    def generate_with_adapter(prompt: str, max_length: int = 150):
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        generated_tokens = input_ids.tolist()[0]
        model = base_model
        model.eval()
        for step in range(max_length):
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
            logits = adapter.adjust_logits(logits, generated_tokens, step)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break
        return tokenizer.decode(generated_tokens, skip_special_tokens=True)
    prompt = "Write a short poem about the ocean."
    adapted_response = generate_with_adapter(prompt)
    logger.info(f"Adapted Response:\n{adapted_response}")
    wikitext = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    wikitext_texts = wikitext['text']
    wikitext_dataset = ResponseDataset(wikitext_texts, tokenizer, MAX_LENGTH)
    train(base_model, tokenizer, wikitext_dataset, os.path.join(MODEL_SAVE_PATH, "implicit_tuned"), NUM_EPOCHS)
    implicit_tuned_model = AutoModelForCausalLM.from_pretrained(os.path.join(MODEL_SAVE_PATH, "implicit_tuned"))
    test_instructions = ["Summarize the plot of 'Pride and Prejudice'.", "Translate to French: 'Hello, how are you?'", "Solve the equation x^2 - 4 = 0."]
    implicit_tuned_model.to(device).eval()
    for instr in test_instructions:
        input_ids = tokenizer.encode(instr, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = implicit_tuned_model.generate(input_ids=input_ids, max_length=150, num_beams=5, early_stopping=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Instruction: {instr}\nResponse: {response}\n")
    evaluation_data = []
    for entry in instructions:
        if 'instruction' in entry and 'response' in entry:
            evaluation_entry = {
                'instruction': entry['instruction'],
                'correct_response': entry['response'],
                'distractor_responses': random.sample(response_texts, 3) if len(response_texts) >= 3 else response_texts
            }
            evaluation_data.append(evaluation_entry)
    logger.info("Evaluating Base Model...")
    evaluate_response_ranking(base_model, tokenizer, evaluation_data)
    logger.info("Evaluating Response-Tuned Model...")
    evaluate_response_ranking(response_tuned_model, tokenizer, evaluation_data)
    logger.info("Evaluating Single-Task Fine-tuned Model...")
    evaluate_response_ranking(single_task_model, tokenizer, evaluation_data)

if __name__ == "__main__":
    main()
