from transformers import BertTokenizer, TFBertForSequenceClassification, create_optimizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import re
import numpy as np
from natasha import Doc, Segmenter, NewsMorphTagger, NewsEmbedding

# Путь к датасету
DATASET_PATH = r'D:\PyCharm Community Edition 2024.1.4\24HACK\TIDIHack\task2\datasets\Dataset_labeled.csv'

# Загрузка данных
def load_data(dataset_path):
    data = pd.read_csv(dataset_path)
    data = data[['comment', 'toxic']]
    data.dropna(inplace=True)
    return data

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

# Подготовка данных с помощью BERT Tokenizer
def prepare_data_with_bert(data, tokenizer, max_len=128):
    encodings = tokenizer(
        data['comment'].tolist(),
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors='tf'
    )
    return encodings

if __name__ == '__main__':
    # Загрузка и предобработка данных
    data = load_data(DATASET_PATH)
    data['comment'] = data['comment'].apply(preprocess_text)

    # Подготовка данных
    MAX_LEN = 128
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = prepare_data_with_bert(data, tokenizer, max_len=MAX_LEN)

    X = encodings['input_ids'].numpy()
    X_attention_masks = encodings['attention_mask'].numpy()
    y = data['toxic'].astype(int).values

    # Проверка форм данных
    print(f"Shapes: X={X.shape}, X_attention_masks={X_attention_masks.shape}, y={y.shape}")

    # Разделение данных
    X_train, X_test, X_mask_train, X_mask_test, y_train, y_test = train_test_split(
        X, X_attention_masks, y, test_size=0.2, random_state=42
    )

    # Проверка форм данных после разделения
    print(f"Train/Test split shapes: X_train={X_train.shape}, y_train={y_train.shape}")

    # Создание модели BERT
    model = TFBertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )

    batch_size = 32
    num_train_steps = (len(X_train) // batch_size) * 5  # количество эпох: 3
    optimizer, _ = create_optimizer(
        init_lr=5e-5,
        num_train_steps=num_train_steps,
        num_warmup_steps=0
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',  # Используем явное указание функции потерь
        metrics=['accuracy']
    )

    # Обучение модели
    history = model.fit(
        {'input_ids': X_train, 'attention_mask': X_mask_train},
        y_train,
        validation_data=(
            {'input_ids': X_test, 'attention_mask': X_mask_test},
            y_test
        ),
        epochs=5,
        batch_size=batch_size
    )

    # Тестирование и оценка
    y_pred_logits = model.predict({'input_ids': X_test, 'attention_mask': X_mask_test})['logits']
    y_pred = (y_pred_logits.argmax(axis=-1)).astype('int32')
    print(classification_report(y_test, y_pred))

    # Сохранение модели
    model.save_pretrained('./models/bert_toxic_model')
    tokenizer.save_pretrained('./models/bert_tokenizer')
    print("Модель и токенизатор успешно сохранены!")
