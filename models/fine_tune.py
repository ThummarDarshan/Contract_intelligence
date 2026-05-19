import logging
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

logger = logging.getLogger(__name__)


def fine_tune_cuad():
    """Fine-tunes legal-roberta-base on the CUAD (Contract Understanding Atticus Dataset) for legal QA."""
    model_name = "saibo-creator/legal-roberta-base"
    
    logger.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    logger.info("Loading CUAD dataset...")
    dataset = load_dataset("theatticusproject/cuad-qa", trust_remote_code=True)
    
    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=512,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            if len(answer["answer_start"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue
                
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    logger.info("Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=50,
        remove_columns=dataset["train"].column_names,
    )

    train_subset = tokenized_datasets["train"].select(range(min(10000, len(tokenized_datasets["train"]))))
    test_subset  = tokenized_datasets["test"].select(range(min(1000,  len(tokenized_datasets["test"]))))

    logger.info("Setting up Trainer...")
    training_args = TrainingArguments(
        output_dir="./fine_tuned_legal_roberta",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        save_steps=200,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_subset,
        eval_dataset=test_subset,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training complete. Saving final model...")
    trainer.save_model("./fine_tuned_legal_roberta")
    tokenizer.save_pretrained("./fine_tuned_legal_roberta")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fine_tune_cuad()
