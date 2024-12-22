import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Путь к датасету
DATASET_PATH = r'D:\PyCharm Community Edition 2024.1.4\24HACK\TIDIHack\task2\datasets\Dataset_labeled.csv'

# Загрузка данных
def load_data(dataset_path):
    data = pd.read_csv(dataset_path)
    data = data[['comment', 'toxic']]
    data.dropna(inplace=True)
    return data

# Предобработка текста
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Удаление пунктуации
    text = re.sub(r'\d+', '', text)  # Удаление чисел
    text = text.lower()  # Приведение к нижнему регистру
    return text

# Подготовка данных
def prepare_data(data, max_words=10000, max_len=100):
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(data['comment'])
    sequences = tokenizer.texts_to_sequences(data['comment'])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded_sequences, tokenizer

# Создание модели
def build_model(vocab_size, embedding_dim=64, input_length=100):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=input_length),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Основная часть
if __name__ == '__main__':
    #Загрузка и предобработка данных
    data = load_data(DATASET_PATH)
    data['comment'] = data['comment'].apply(preprocess_text)

    #Подготовка данных
    MAX_WORDS = 10000
    MAX_LEN = 100
    X, tokenizer = prepare_data(data, max_words=MAX_WORDS, max_len=MAX_LEN)
    y = data['toxic'].values

    #Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Создание модели
    model = build_model(vocab_size=MAX_WORDS, embedding_dim=64, input_length=MAX_LEN)

    #Обучение модели
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=12,
        batch_size=32,
        callbacks=[early_stopping]
    )
    # Сохраняем токенизатор
    with open('./models/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #Сохранение модели
    model.save('./models/toxic_model.h5')
    print("Модель успешно сохранена!")
