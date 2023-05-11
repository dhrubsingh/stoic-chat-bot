This is the repository behind my Stoic Chat Bot, which has been trained on GPT-2. To boil down the essence of this project, I followed essentially the typical framework of creating an AI-chatbot: (1) gather the source data from which I was going to train the model, (2) preprocess the data, (3) and then training the data, (4) creating a simple user interface. I will describe each of these steps in detail so that other reseachers can replicate a similar project in the future.

Firstly, to describe why I chose to use GPT-2 instead of other LLMs is for the following: GPT-3 and OpenAI have recently been causing issues with accepting payments for their API service. Furthermore, for a smaller project like this, the cost component of using GPT-3 isn't as valuable as just documenting the entire process of creating a chat bot, as using GPT-3 would work exactly the way I have structured my code right now, with the same flow but just a different model to train on.

For the first step of this project, I needed to gather the source data from which I was going to train the model. Since I was creating a Stoic Chat bot, I web scraped two key stoic books: Discourses and Meditations. I found both of these from Project Gutenberg, which is an extremely useful online library that digitizes a lot of key philsophical texts. I then generated two texts files, *discourses.txt* and *meditations.txt* to contain the scraped contents of these scripts. The script *scraper.py* is the code I used to scrape these two books.

Secondly, I then preprocessed the text files using the script *preprocess.py* so that it could be accurately fed into the GPT-2 model. This step invovled a lot of tokenization of the data, since that is the way GPT-2 can understand the input data. I specifically broke up the text into sequences of 1024 characters (tokens) each, and then sequentially defined three more texts files: a train_file, *book_name_train.txt*, a value file, *book_name_val.txt*, and a test_file *book_name_test.txt". These three files are the key input components for training the model in the next step.

Thirdy, I finally trained the model from the tokenized files as mentioned before. This is done in the *train.py* script, which takes these tokenized files and then trains the model. On average, training the model on a big text file like *Meditations* and *Discourses* took me around 45 minutes per file. Because GitHub can't accept 100 mb or larger files, I can't actually push my model into my repo (a problem I'll talk about later).

Furthermore, after I finished training the model on the input files, I 