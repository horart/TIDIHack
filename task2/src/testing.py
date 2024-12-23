from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import pandas as pd
import re
from natasha import Doc, Segmenter, NewsMorphTagger, NewsEmbedding

# Предобработка текста
segmenter = Segmenter()
morph_tagger = NewsMorphTagger(NewsEmbedding())

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Удаление пунктуации
    text = re.sub(r'\d+', '', text)  # Удаление чисел
    text = text.lower()  # Приведение к нижнему регистру
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    lemmas = [token.lemma for token in doc.tokens if token.pos not in {'PUNCT', 'NUM'} and token.lemma is not None]
    return ' '.join(lemmas)

# Функция для загрузки данных из .txt файла
def load_unlabeled_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return pd.DataFrame({'comment': [line.strip() for line in lines]})

# Разметка данных с помощью обученной модели
def label_data_with_model(model, tokenizer, data, max_len=128):
    # Предобработка текста
    data['comment'] = data['comment'].apply(preprocess_text)

    # Токенизация
    encodings = tokenizer(
        data['comment'].tolist(),
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors='tf'
    )

    # Предсказание
    predictions = model.predict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask']
    })['logits']
    data['toxic'] = tf.nn.softmax(predictions, axis=-1).numpy()[:, 1] > 0.5  # Метка токсичности
    return data

if __name__ == '__main__':
    # Пути к модели, токенизатору и данным
    MODEL_PATH = './models/bert_toxic_model'
    TOKENIZER_PATH = './models/bert_tokenizer'
    # поменять пути
    UNLABELED_DATA_PATH = 'D:/PyCharm Community Edition 2024.1.4/24HACK/TIDIHack/task2/datasets/Dataset_2_unlabeled.txt'
    OUTPUT_PATH = 'D:/PyCharm Community Edition 2024.1.4/24HACK/TIDIHack/task2/datasets/labeled_data.csv'

    # Загрузка модели и токенизатора
    model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH)  # Загрузка модели из .h5 файла
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)

    # Загрузка и разметка неразмеченных данных
    data = load_unlabeled_data(UNLABELED_DATA_PATH)
    labeled_data = label_data_with_model(model, tokenizer, data)

    # Сохранение размеченных данных
    labeled_data.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
    print(f"Размеченные данные сохранены в {OUTPUT_PATH}!")
