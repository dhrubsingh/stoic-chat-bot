import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load fine-tuning data
train_file = 'meditations_ft.txt'

# Load trained model
model = GPT2LMHeadModel.from_pretrained('meditations_model')

# Create datasets
tokenizer = GPT2Tokenizer.from_pretrained('meditations_model')
with open(train_file, 'r') as f:
    text = f.read()
sequences = [text[i:i+1024] for i in range(0, len(text), 1024)]
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=1024,
)
val_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
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
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_steps=100,
    save_steps=500,
    logging_steps=100,
    save_total_limit=10,
    learning_rate=2e-5  # Adjust the learning rate if needed
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# Fine-tune model
trainer.train()

# Save fine-tuned model
model.save_pretrained('meditations_model')
tokenizer.save_pretrained('meditations_model')
