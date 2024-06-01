import yaml
import torch
from trl import SFTTrainer
from transformers import TrainingArguments

from config import load_config
from prompts import alpaca_prompt
from model_prep import prepare_model
from data_prep import load_and_format_dataset

class ModelTrainer:
    def __init__(self, yaml_config_path, output_dir='outputs'):
        self.config = load_config()
        self.yaml_config = self.load_yaml_config(yaml_config_path)
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None
        self.dataset = None

    def load_yaml_config(self, yaml_file):
        with open(yaml_file, 'r') as file:
            return yaml.safe_load(file)

    def prepare_model_and_dataset(self):
        model_name = self.yaml_config['model']['name']
        dataset_repo_name = self.yaml_config['dataset']['repo_name']
        max_seq_length = self.yaml_config['model']['max_seq_length']
        prompt = alpaca_prompt

        self.model, self.tokenizer = prepare_model(model_name, max_seq_length)
        self.dataset = load_and_format_dataset(
            self.config['HUB_TOKEN'],
            dataset_repo_name,
            self.tokenizer.eos_token,
            prompt
        )

        self.tokenizer.padding_side = self.yaml_config['tokenizer']['padding_side']

    def train_model(self):
        self.prepare_model_and_dataset()
        
        training_args = TrainingArguments(
            per_device_train_batch_size=self.yaml_config['training']['per_device_train_batch_size'],
            gradient_accumulation_steps=self.yaml_config['training']['gradient_accumulation_steps'],
            warmup_steps=self.yaml_config['training']['warmup_steps'],
            num_train_epochs=self.yaml_config['training']['num_train_epochs'],
            max_steps=self.yaml_config['training']['max_steps'],
            logging_steps=self.yaml_config['training']['logging_steps'],
            save_steps=self.yaml_config['training']['save_steps'],
            learning_rate=self.yaml_config['training']['learning_rate'],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim=self.yaml_config['training']['optim'],
            weight_decay=self.yaml_config['training']['weight_decay'],
            lr_scheduler_type=self.yaml_config['training']['lr_scheduler_type'],
            # seed=self.yaml_config['training']['seed'],
            output_dir=self.yaml_config['training']['output_dir'],
            ddp_find_unused_parameters=self.yaml_config['training']['ddp_find_unused_parameters'],
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            dataset_text_field="text",
            max_seq_length=self.yaml_config['model']['max_seq_length'],
            dataset_num_proc=self.yaml_config['dataset']['num_proc'],
            packing=self.yaml_config['dataset']['packing'],
            args=training_args,
        )

        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory}GB.")
        print(f"{start_gpu_memory}GB of memory reserved.")

        trainer_stats = trainer.train()
        return trainer_stats

# Example usage
if __name__ == "__main__":
    from model_save import ModelManager

    model_config_path = "model_config/config_llama3_openko_8b.yaml"
    trainer = ModelTrainer(model_config_path)
    trainer.train_model()

    print("Training complete. Pushing model to Hugging Face Hub.")

    base_model_name = "beomi/Llama-3-Open-Ko-8B"
    finetuned_model_name = "Llama3-Open-Ko-8B-Instruct-toeic4all"
    quantization_method = 'q8_0'    # "f16", "q8_0", "q4_k_m", "q5_k_m"
    
    manager = ModelManager()
    manager.load_model_from_checkpoint()
    manager.push_model_to_hub(
        base_model_name,
        finetuned_model_name
        )
    manager.push_model_to_hub_gguf(
        finetuned_model_name,
        quantization_method=quantization_method
        )
