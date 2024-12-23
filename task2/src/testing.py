from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle
import pandas as pd
from sklearn.metrics import f1_score

# Загрузка сохраненного токенизатора
with open('./models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
# Загрузка модели
model = load_model('./models/toxic_model.h5')

# Проверка структуры модели
model.summary()

# Предполагается, что токенайзер был обучен на данных при обучении модели
# Если токенайзер был сохранен, его нужно загрузить. Например:


# В случае, если токенизатор не был сохранен, нужно использовать тот же код для его обучения:
# Пример текста для предсказания
sample_text = ["И ещё один, какая же русня дегенераты пиздец просто.", " "]

# Подготовка данных
# Если токенизатор обучался раньше, его необходимо использовать:
# tokenizer = Tokenizer(num_words=10000)  # или загрузить ранее обученный токенизатор
# tokenizer.fit_on_texts(sample_text)  # Обучаем токенизатор на тестовых данных (для примера)

# Преобразуем текст в последовательности и применяем паддинг
sequences = tokenizer.texts_to_sequences(sample_text)
padded_sequences = pad_sequences(sequences, maxlen=100)

# Прогнозирование с моделью
predictions = model.predict(padded_sequences)

# Вывод предсказаний
print(predictions)

DATASET_PATH = '../datasets/Dataset_labeled.csv'

data = pd.read_csv(DATASET_PATH)
data = data[['comment', 'toxic']].iloc[int(len(data)/2):, :]
data.dropna(inplace=True)
sequences = tokenizer.texts_to_sequences(data['comment'])
padded_sequences = pad_sequences(sequences, maxlen=100)
predictions = tf.round(model.predict(padded_sequences))
print(f1_score(data['toxic'], predictions))