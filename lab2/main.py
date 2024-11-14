import os
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pdfplumber
import time
from collections import Counter
from nltk.corpus import stopwords
import nltk
from g4f.client import Client

os.makedirs("result", exist_ok=True)
size = 3
min_freq = 0.0001

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text


def create_probability_model(word_list):
    probabilities = {word: freq for word, freq in word_list.items() if freq >= 2}
    total_count = sum(probabilities.values())
    probabilities = {word: freq / total_count for word, freq in probabilities.items()}
    return probabilities


def short_analyse_text(text):
    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word.isalpha() and len(word) <= size]

    words = Counter(words)
    words = create_probability_model(words)

    russian_model_words = [word for word in stopwords.words('russian') if len(word) <= size]
    italian_model_words = [word for word in stopwords.words('italian') if len(word) <= size]

    probability_r = 1
    probability_i = 1

    for mod_word in russian_model_words:
        prob_r = words.get(mod_word, min_freq)
        probability_r *= prob_r

    for mod_word in italian_model_words:
        prob_i = words.get(mod_word, min_freq)
        probability_i *= prob_i

    if probability_r > probability_i:
        return [word for word in words if len(word) <= size], 'ru'
    else:
        return [word for word in words if len(word) <= size], 'it'


def freq_analyze_text(text: str):
    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word.isalpha()]
    word_freq = Counter(words)

    russian_words = stopwords.words('russian')
    italian_words = stopwords.words('italian')

    language_word_freq = ''

    for word, freq in word_freq.most_common(10):
        if word in russian_words:
            language_word_freq = 'ru'
            break
        elif word in italian_words:
            language_word_freq = 'it'
            break
        else:
            language_word_freq = "Не удалось определить язык"
    return word_freq, language_word_freq


def nn_analyse(text):
    client = Client()
    question = f"Определи язык этого текста: {text[:450]} \n Если русский - ru, если итальянский - it"
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content


def record_result(file_name, word_freq, short_words, language_word_freq, language_short_word, time, nn_result):
    result_path = f"result/{file_name}.txt"
    with open(result_path, "w") as result_file:
        result_file.write("Результаты анализа PDF\n")
        result_file.write("\nЧастотный анализ (Топ-10 слов):\n")
        for word, freq in word_freq:
            result_file.write(f"{word}: {freq}\n")
        result_file.write(f"Язык на основе частотного анализа - {language_word_freq}\n")
        result_file.write(f"\nКороткие слова (<= {size} символов):\n")
        for word in short_words:
            result_file.write(f"{word}\n")
        result_file.write(f'Язык на основе анализа коротких слов - {language_short_word}\n')
        result_file.write(f"\nЯзык на основе нейросетевого анализа - {nn_result}\n")
        result_file.write(f"\nВремя затраченное на анализ текста - {time}")


@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_pdf(request: Request, file: UploadFile = File(...)):
    file_name = file.filename
    text = extract_text_from_pdf(file.file)
    start = time.time()
    word_freq, language_word_freq = freq_analyze_text(text)
    short_words, language_short_word = short_analyse_text(text)
    nn_result = nn_analyse(text)
    finish = time.time()-start
    record_result(file_name, word_freq.most_common(10), short_words[:10], language_word_freq, language_short_word, finish, nn_result)

    return templates.TemplateResponse("result.html", {
        "language_word_freq": language_word_freq,
        "language_short_word": language_short_word,
        "request": request,
        "word_freq": word_freq.most_common(10),
        "short_words": short_words[:10],
        "nn_result": nn_result,
        "file_name": file_name,
        "time": finish,
        "size": str(size)
    })