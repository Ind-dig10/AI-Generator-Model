from flask import Flask, request, render_template, redirect, session, url_for, jsonify, send_file
import pandas as pd
import os
import json
from services.train import Train
from config import ConnectionString
import psycopg2
import pickle
import io
from joblib import load


app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file and file.filename.endswith('.csv'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        data = pd.read_csv(file_path)
        session['columns'] = data.columns.tolist()  # Сохраняем колонки в сессии
        session['file_path'] = file_path  # Сохраняем путь к файлу
        return redirect(url_for('info'))
    return "Ошибка: загрузите CSV файл."


@app.route('/info', methods=['GET', 'POST'])
def info():
    if request.method == 'POST':
        target_column = request.form['target_column']
        session['target_column'] = target_column  # Сохраняем выбранную метрику в сессии
        return redirect(url_for('train'))

    columns = session.get('columns', [])
    return render_template('info.html', columns=columns)


@app.route('/train')
def train():
    file_path = session.get('file_path')
    target_column = session.get('target_column')
    results = Train(file_path, target_column).execute()

    return render_template('result.html', results=results)


# Метод для загрузки лучшей модели
@app.route('/download_model', methods=['GET'])
def download_model():
    try:
        # Подключение к базе данных
        conn = psycopg2.connect(**ConnectionString)
        cursor = conn.cursor()

        # Запрос на получение лучшей модели и скалера
        cursor.execute("SELECT model_name, model_data, scaler_data FROM best_models ORDER BY mse LIMIT 1")
        result = cursor.fetchone()

        if result is None:
            return jsonify({"error": "Модель не найдена в базе данных."}), 404

        model_name, model_data, scaler_data = result

        # Десериализация модели и скалера
        model = pickle.loads(model_data)
        scaler = pickle.loads(scaler_data)

        # Создание объекта BytesIO для модели
        model_file = io.BytesIO()
        pickle.dump(model, model_file)
        model_file.seek(0)

        # Создание объекта BytesIO для скалера
        scaler_file = io.BytesIO()
        pickle.dump(scaler, scaler_file)
        scaler_file.seek(0)

        # Создание имени для архива
        archive_name = f"{model_name}.zip"

        # Отправка архива с моделью и скалером
        from zipfile import ZipFile
        from io import BytesIO

        # Создание ZIP файла в памяти
        zip_buffer = BytesIO()
        with ZipFile(zip_buffer, 'w') as zip_file:
            zip_file.writestr(f"{model_name}.pkl", model_file.getvalue())
            zip_file.writestr(f"{model_name}_scaler.pkl", scaler_file.getvalue())

        zip_buffer.seek(0)

        # Отправка ZIP файла пользователю
        return send_file(zip_buffer, as_attachment=True, download_name=archive_name, mimetype="application/zip")

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()


@app.route('/test_model_page', methods=['GET'])
def test_model_page():
    columns = session.get('columns', [])
    target_column = session.get('target_column')
    columns.remove(target_column)
    return render_template('test_model.html', columns=columns)


@app.route('/test_model', methods=['POST'])
def test_model():
    try:
        # Получаем параметры из формы
        input_data = {}
        for column in request.form:
            input_data[column] = float(request.form[column])  # Преобразуем в float или другой нужный тип

        # Подключение к базе данных и загрузка модели
        conn = psycopg2.connect(**ConnectionString)
        cursor = conn.cursor()

        # Запрос на получение лучшей модели (по минимальному MSE)
        cursor.execute("SELECT model_data, scaler_data FROM best_models ORDER BY mse LIMIT 1")
        result = cursor.fetchone()

        if result is None:
            return jsonify({"error": "Модель не найдена в базе данных."}), 404

        model_data, scaler_data = result

        # Загрузка модели и масштабировщика
        model = pickle.loads(model_data)
        scaler = pickle.loads(scaler_data)

        # Преобразуем входные данные в DataFrame
        input_df = pd.DataFrame([input_data])

        # Проверка на наличие пустых значений в данных
        if input_df.isnull().any().any():
            return jsonify({"error": "Входные данные содержат пропущенные значения."}), 400

        # Масштабирование входных данных
        X_new_scaled = scaler.transform(input_df)

        prediction = model.predict(X_new_scaled)

        return render_template('prediction_result.html', prediction=prediction[0])

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        conn.close()



if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    app.run(debug=True)


