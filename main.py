import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical  # Bu satırı ekleyin

# Veri yükleme
train = pd.read_csv(r"C:\Users\ugrkr\OneDrive\Masaüstü\obesity\train.csv")
test = pd.read_csv(r"C:\Users\ugrkr\OneDrive\Masaüstü\obesity\test.csv")
sub = pd.read_csv(r"C:\Users\ugrkr\OneDrive\Masaüstü\obesity\sample_submission.csv")
print(train.head())
print(train.info())

# Özellikler ve hedef değişkenlerin ayrılması
X = train.drop(columns=["NObeyesdad", "id"], axis=1)
y = train["NObeyesdad"]

# Hedef değişkenin etiketlenmesi
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Sayısal ve kategorik sütunları ayırma
categorical_cols = X.select_dtypes(include="object").columns
numerical_cols = X.select_dtypes(include=["int", "float"]).columns

# Kategorik verilerde baskın sınıf kontrolü
delete_list = []
for col in categorical_cols:
    dominant_v = X[col].value_counts(normalize=True)
    if dominant_v.max() > 0.85:
        delete_list.append(col)
        print(f"{col} will be deleted due to dominant class.")

X.drop(columns=delete_list, inplace=True)
categorical_cols = X.select_dtypes(include="object").columns
numerical_cols = X.select_dtypes(include=["int", "float"]).columns

# Sayısal ve kategorik sütunlar için pipeline oluşturma
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# ColumnTransformer oluşturma
preprocess = ColumnTransformer(transformers=[
    ("num", numerical_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# Eğitim ve test verilerini ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)

# Keras sinir ağı modelini oluşturma
X_train_nn = preprocess.fit_transform(X_train)
X_test_nn = preprocess.transform(X_test)

y_encoded_nn = to_categorical(y_train)
y_test_nn = to_categorical(y_test)

model = Sequential([
    Dense(32, activation="relu", input_shape=(X_train_nn.shape[1],)),
    Dense(16, activation="relu"),
    Dense(y_encoded_nn.shape[1], activation="softmax")
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history = model.fit(X_train_nn, y_encoded_nn, epochs=10, batch_size=32, validation_split=0.25)

# Eğitim ve doğrulama başarı oranlarını alma
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

print("Training Accuracy:", train_accuracy[-1])
print("Validation Accuracy:", val_accuracy[-1])

test = pd.read_csv(r"C:\Users\ugrkr\OneDrive\Masaüstü\obesity\test.csv")
X_test_final = test.drop(columns=["id"], axis=1)

X_test_final = X_test_final.drop(columns=delete_list, errors='ignore')
X_test_final_processed = preprocess.transform(X_test_final)
keras_predictions = model.predict(X_test_final_processed)
keras_predictions = np.argmax(keras_predictions, axis=1)
keras_labels = label_encoder.inverse_transform(keras_predictions)
submission = pd.DataFrame({
    'id': test['id'],
    'NObeyesdad': keras_labels  # veya 'NObeyesdad': xgb_predictions, eğer XGBoost tahminleri kullanıyorsanız
})

submission.to_csv(r'C:\Users\ugrkr\OneDrive\Masaüstü\obesity\submission_obestity.csv', index=False)
print("Submission file has been saved.")
