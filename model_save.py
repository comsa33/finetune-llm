import os
from unsloth import FastLanguageModel
from config import load_config

class ModelManager:
    def __init__(self, config_path='config.yaml', output_dir='outputs'):
        self.config = load_config(config_path)
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None

    def _get_latest_checkpoint(self):
        checkpoints = [
            int(x.split('-')[-1]) for x in os.listdir(self.output_dir) if 'checkpoint' in x
        ]
        if not checkpoints:
            raise ValueError("No checkpoints found in the output directory.")
        latest_checkpoint = max(checkpoints)
        return f"{self.output_dir}/checkpoint-{latest_checkpoint}"

    def load_model_from_checkpoint(self, max_seq_length=4096, dtype=None, load_in_4bit=True):
        checkpoint_dir = self._get_latest_checkpoint()
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_dir,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit
        )
        return self.model, self.tokenizer

    def push_model_to_hub(self, base_model, huggingface_repo, save_method="merged_16bit"):
        hugginface_token = self.config['HUB_TOKEN']
        self.model.push_to_hub_merged(
            huggingface_repo,
            self.tokenizer,
            save_method=save_method,
            token=hugginface_token,
        )

    def push_model_to_hub_gguf(self, huggingface_repo, quantization_method='q8_0'):
        hugginface_token = self.config['HUB_TOKEN']
        self.model.push_to_hub_gguf(
            huggingface_repo + "-gguf",
            self.tokenizer,
            quantization_method=quantization_method,
            token=hugginface_token,
        )

# Example usage
if __name__ == "__main__":
    manager = ModelManager()
    manager.load_model_from_checkpoint()
    manager.push_model_to_hub("beomi/Llama-3-Open-Ko-8B", "Llama3-Open-Ko-8B-Instruct-ruolee")
    manager.push_model_to_hub_gguf("Llama3-Open-Ko-8B-Instruct-ruolee", quantization_method='q8_0')
