!pip install PyPDF2
!pip install transformers
!pip install torch
!pip install python-telegram-bot==13.7 --force-reinstall
import PyPDF2
import telegram
from telegram import Update
from telegram.ext import CommandHandler, MessageHandler, CallbackContext
from telegram.ext import Filters
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer

# Telegram bot token
TOKEN = '6150996741:AAEFWHba6QkkcCAhvf6-reX3azvih6UbAiY'

# Load the DistilBERT model and tokenizer
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')

# Define the command handler for the /start command
def start(update: Update, context: CallbackContext):
    update.message.reply_text('Welcome To maths formulae! Please send me your question . I am Happy to Assist You.')

# Define the message handler for user's questions
def handle_message(update: Update, context: CallbackContext):
    """Handle user's questions and provide answers."""
    # Get the user's question
    question = update.message.text

    # Read the PDF file and extract the content
    pdf_path = '/content/MathFor.pdf'
    pdf_content = ''
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            pdf_content += page.extract_text()

    # Split the PDF content into smaller chunks
    chunk_size = 512  # Maximum sequence length supported by the model
    chunks = [pdf_content[i:i + chunk_size] for i in range(0, len(pdf_content), chunk_size)]

    # Tokenize the question and each chunk of PDF content
    inputs = []
    for chunk in chunks:
        input_dict = tokenizer.encode_plus(question, chunk, add_special_tokens=True, return_tensors="pt")
        inputs.append(input_dict)

    # Perform question answering inference for each chunk
    answers = []
    for input_dict in inputs:
        input_ids = input_dict["input_ids"]
        attention_mask = input_dict["attention_mask"]
        outputs = model(input_ids, attention_mask=attention_mask)
        
        start_scores = outputs.start_logits.squeeze().detach().numpy()
        end_scores = outputs.end_logits.squeeze().detach().numpy()
        
        start_index = start_scores.argmax()
        end_index = end_scores.argmax()
        
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())
        answer = tokenizer.convert_tokens_to_string(tokens[start_index:end_index+1])
        confidence_score = (start_scores[start_index] + end_scores[end_index]) / 2
        answers.append((answer, confidence_score))

    # Sort answers by confidence score in descending order
    answers.sort(key=lambda x: x[1], reverse=True)

    # Get the best answer and its confidence score
    best_answer = answers[0][0]
    best_answer1 = answers[1][0]
    confidence_score = answers[0][1]

    # Send the best answer back to the user
    response = f"formula: {best_answer}"
    update.message.reply_text(response)
    response1 = f"Description: {best_answer1}"
    update.message.reply_text(response1)

# Create the Telegram bot and dispatcher
bot = telegram.Bot(token=TOKEN)
updater = telegram.ext.Updater(bot=bot, use_context=True)
dispatcher = updater.dispatcher

# Register the handlers
dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

# Start the bot
updater.start_polling()
updater.idle()