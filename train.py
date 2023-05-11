import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

def train_gpt2_model(train_file, val_file, model_output_dir):
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Create datasets
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=1024,
    )
    val_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=val_file,
        block_size=1024,
    )

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # Set training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_steps=100,
        save_steps=500,
        logging_steps=100,
        save_total_limit=10,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train model
    trainer.train()

    # Save model and tokenizer
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

# train on discourses
train_file = 'discourses_ft_train.txt'
val_file = 'discourses_ft_val.txt'
model_output_dir = 'meditations_model'

train_gpt2_model(train_file, val_file, model_output_dir)