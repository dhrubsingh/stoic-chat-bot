import re

def preprocess_text(text, book_name):
    # Preprocess text
    text = text.lower().strip()  # Convert to lowercase and remove leading/trailing whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\n', ' ', text)  # Replace line breaks with spaces

    # Split text into sequences of length 1024
    seq_length = 1024
    sequences = []
    for i in range(0, len(text), seq_length):
        seq = text[i:i+seq_length]
        sequences.append(seq)

    # Save sequences to text files for training, validation, and testing
    train_file = f"{book_name}_train.txt"
    val_file = f"{book_name}_val.txt"
    test_file = f"{book_name}_test.txt"

    train_size = int(len(sequences) * 0.8)
    val_size = int(len(sequences) * 0.1)

    with open(train_file, 'w') as f:
        f.write('\n'.join(sequences[:train_size]))

    with open(val_file, 'w') as f:
        f.write('\n'.join(sequences[train_size:train_size+val_size]))

    with open(test_file, 'w') as f:
        f.write('\n'.join(sequences[train_size+val_size:]))


# preprocess Discourses
with open('discourses_ft.txt', 'r') as f:
    text = f.read()

preprocess_text(text, 'discourses_ft')
