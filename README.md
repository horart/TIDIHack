# Хакатон: Анализ токсичных комментариев  

## Описание  
Этот хакатон посвящен созданию и тестированию модели машинного обучения, которая определяет токсичность текста. Участники будут тренировать модель на одном размеченном датасете, а затем проверять ее на другом датасете без разметки. После завершения работы предоставится истинная разметка второго датасета для оценки точности модели.  

## Цели  
1. Погрузиться в работу с текстовыми данными и изучить их обработку.  
2. Освоить основные этапы создания моделей машинного обучения.  
3. Развить навыки тестирования и оценки качества моделей.  

## Структура проекта  
1. **Сбор данных:**  
   - Ознакомление с размеченным датасетом для тренировки модели.  
   - Выбор подходящих характеристик (features) для обучения.  
2. **Обучение модели:**  
   - Подготовка данных: очистка текста, преобразование в числовой вид.  
   - Обучение модели машинного обучения.  
3. **Тестирование:**  
   - Применение модели к новому датасету без разметки.  
   - Сравнение предсказаний модели с истинной разметкой после ее предоставления.  
4. **Анализ результатов:**  
   - Вычисление метрик точности, таких как F1-score, precision и recall.  
   - Обсуждение результатов и улучшений.  

## Задания  
1. Загрузите размеченный датасет с Kaggle: [Russian Language Toxic Comments](https://www.kaggle.com/datasets/blackmoon/russian-language-toxic-comments/data).  
2. Проведите базовый анализ данных:  
   - Определите количество строк, типов токсичности и распределение классов.  
3. Обработайте текстовые данные:  
   - Очистите текст (удаление лишних символов, приведение к нижнему регистру).  
   - Преобразуйте текст в числовое представление (например, TF-IDF или word embeddings).  
4. Обучите модель на размеченных данных.  
5. Примените модель к новому датасету и сохраните результаты.  
6. После предоставления разметки оцените точность модели.  

## Необходимые инструменты  
- **Язык программирования:** Python  
- **Инструменты:** Jupyter Notebook  
- **Библиотеки:**  
  - `pandas`  
  - `scikit-learn`  
  - `nltk`  
  - `matplotlib`  

## Критерии успеха  
- Разработанная модель должна уметь определять токсичность комментариев.  
- Точность модели на тестовом наборе данных должна быть как можно выше (ориентировочно F1-score > 0.7).  

## Полезные ресурсы  
- [Документация scikit-learn](https://scikit-learn.org/stable/).  
- [Kaggle Dataset: Russian Language Toxic Comments](https://www.kaggle.com/datasets/blackmoon/russian-language-toxic-comments/data).  
- [Руководство по работе с текстовыми данными: nltk.org](https://www.nltk.org).  

## Сдача работы  
1. Сохраните Jupyter Notebook с результатами.  
2. Создайте отчет с кратким описанием подхода и метрик.  
3. Отправьте проект на проверку организаторам хакатона.  
