from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.feature_selection import RFE
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
#import optuna
import pickle
import psycopg2
from config import ConnectionString

class Train():
    def __init__(self, dataset, target_column):
        self.data = pd.read_csv(dataset)
        self.target_column = target_column

        # Задание целевой переменной
        self.X = self.data.drop(columns=[target_column])
        self.y = self.data[target_column]

        # Разделение на тренировочную и тестовую выборки
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Нормализация данных
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # Словарь для хранения результатов моделей
        self.results_mse = {}
        self.models = {}

    def execute(self):
        lr_results = self.linear_regression()
        rf_results = self.random_forest()
        gb_results = self.gradien_boosting()
        nn_results = self.neural_network()
        xgb_results = self.xgboost()

        # Определение модели с наименьшим MSE
        best_model_name = min(self.results_mse, key=self.results_mse.get)
        best_model_mse = self.results_mse[best_model_name]
        best_model = self.models[best_model_name]

        results = {
            'Linear Regression': lr_results,
            'Random Forest': rf_results,
            'Gradient Boosting': gb_results,
            'Neural Network': nn_results,
            "XGBRegressor Gradient Boosting" : xgb_results,
            'Лучшая модель': best_model_name,
            'Лучшее MSE': best_model_mse
        }

        self.save_model_to_db(best_model, best_model_name, best_model_mse)
        return results

    # Сохранить модель в бд
    def save_model_to_db(self, model, model_name, mse):
        model_data = pickle.dumps(model)
        scaler_data = pickle.dumps(self.scaler)

        conn = psycopg2.connect(**ConnectionString)
        cursor = conn.cursor()

        # Создаем таблицу, если её нет
        cursor.execute("""
                    CREATE TABLE IF NOT EXISTS best_models (
                        model_name VARCHAR(255) PRIMARY KEY,
                        mse FLOAT,
                        model_data BYTEA,
                        scaler_data BYTEA
                    )
                """)

        # Перезаписываем запись, если модель с таким именем уже существует
        cursor.execute("""
                    INSERT INTO best_models (model_name, mse, model_data, scaler_data)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (model_name)
                    DO UPDATE SET
                        mse = EXCLUDED.mse,
                        model_data = EXCLUDED.model_data
                """, (model_name, mse, model_data, scaler_data))
        conn.commit()
        cursor.close()
        conn.close()


    # Линейная регрессия
    def linear_regression(self):
        lr_model = LinearRegression()
        rfe = RFE(lr_model, n_features_to_select=4)
        rfe.fit(self.X_train_scaled, self.y_train)
        lr_predictions = rfe.predict(self.X_test_scaled)

        print("Linear Regression MSE:", mean_squared_error(self.y_test, lr_predictions))
        print("Linear Regression MAE:", mean_absolute_error(self.y_test, lr_predictions))
        print("Linear Regression R2:", r2_score(self.y_test, lr_predictions))

        lr_results = {
            'MSE': mean_squared_error(self.y_test, lr_predictions),
            'MAE': mean_absolute_error(self.y_test, lr_predictions),
            'R2': r2_score(self.y_test, lr_predictions)
        }

        self.results_mse['Linear Regression'] = mean_squared_error(self.y_test, lr_predictions)
        self.models['Linear Regression'] = rfe

        return lr_results


    # Random Forest
    def random_forest(self):
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train_scaled, self.y_train)
        rf_predictions = rf_model.predict(self.X_test_scaled)

        # Оценка Random Forest
        print("Random Forest MSE:", mean_squared_error(self.y_test, rf_predictions))
        print("Random Forest MAE:", mean_absolute_error(self.y_test, rf_predictions))
        print("Random Forest R2:", r2_score(self.y_test, rf_predictions))

        rf_results = {
            'MSE': mean_squared_error(self.y_test, rf_predictions),
            'MAE': mean_absolute_error(self.y_test, rf_predictions),
            'R2': r2_score(self.y_test, rf_predictions)
        }

        self.results_mse['Random Forest'] = mean_squared_error(self.y_test, rf_predictions)
        self.models['Random Forest'] = rf_model

        return rf_results


    # Gradient Boosting
    def gradien_boosting(self):
        gb_model = GradientBoostingRegressor(random_state=42)
        gb_model.fit(self.X_train_scaled, self.y_train)
        gb_predictions = gb_model.predict(self.X_test_scaled)

        # Оценка Gradient Boosting
        print("Gradient Boosting MSE:", mean_squared_error(self.y_test, gb_predictions))
        print("Gradient Boosting MAE:", mean_absolute_error(self.y_test, gb_predictions))
        print("Gradient Boosting R2:", r2_score(self.y_test, gb_predictions))

        # Оценка Gradient Boosting
        gb_results = {
            'MSE': mean_squared_error(self.y_test, gb_predictions),
            'MAE': mean_absolute_error(self.y_test, gb_predictions),
            'R2': r2_score(self.y_test, gb_predictions)
        }
        self.results_mse['Gradient Boosting'] = mean_squared_error(self.y_test, gb_predictions)
        self.models['Gradient Boosting'] = gb_model

        return gb_results


    # Нейронная сеть
    def neural_network(self):
        nn_model = Sequential()
        nn_model.add(Dense(64, input_dim=self.X_train_scaled.shape[1], activation='relu'))
        nn_model.add(Dense(32, activation='relu'))
        nn_model.add(Dense(1))  # выходной слой для регрессии
        nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Обучение нейронной сети
        nn_model.fit(self.X_train_scaled, self.y_train, epochs=50, batch_size=10, verbose=1)
        nn_predictions = nn_model.predict(self.X_test_scaled)

        # Оценка нейронной сети
        print("Neural Network MSE:", mean_squared_error(self.y_test, nn_predictions))
        print("Neural Network MAE:", mean_absolute_error(self.y_test, nn_predictions))
        print("Neural Network R2:", r2_score(self.y_test, nn_predictions))

        nn_results = {
                'MSE': mean_squared_error(self.y_test, nn_predictions),
                'MAE': mean_absolute_error(self.y_test, nn_predictions),
                'R2': r2_score(self.y_test, nn_predictions)
        }

        self.results_mse['Neural Network'] = mean_squared_error(self.y_test, nn_predictions)
        self.models['Neural Network'] = nn_model

        return nn_results

    # Градиентный бустинг xgboost
    def xgboost(self):
        xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=42)
        xgb_model.fit(self.X_train_scaled, self.y_train)
        xgb_predictions = xgb_model.predict(self.X_test_scaled)
        self.results_mse['XGBoost'] = mean_squared_error(self.y_test, xgb_predictions)
        self.models['XGBoost'] = xgb_model

        xg_results = {
            'MSE': mean_squared_error(self.y_test, xgb_predictions),
            'MAE': mean_absolute_error(self.y_test, xgb_predictions),
            'R2': r2_score(self.y_test, xgb_predictions)
        }
        return xg_results