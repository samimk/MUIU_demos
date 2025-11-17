"""
WiFi Indoor Positioning System - Kompletna Implementacija
=========================================================
Dataset: UJIIndoorLoc (WiFi fingerprinting za pozicioniranje u zgradi)
Cilj: Predviđanje pozicije (zgrada, sprat, koordinate) na osnovu WiFi signala
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import keras
from keras import layers, models, callbacks, regularizers
from keras.utils import to_categorical, plot_model
import warnings
import os
warnings.filterwarnings('ignore')

# Postavljanje seed-a za reproducibilnost
np.random.seed(42)
keras.utils.set_random_seed(42)

# Putanja do CSV fajla sa WiFi podacima
# Ako fajl ne postoji, koristiće se simulirani podaci
WIFI_DATA_FILE = "wifi_collected_data.csv"

print("Keras verzija:", keras.__version__)

# ============================================================================
# 1. UČITAVANJE I EKSPLORACIJA PODATAKA
# ============================================================================

def load_data_from_csv(csv_file):
    """
    Učitavanje podataka iz CSV fajla prikupljenog aplikacijom get_wifi_data.py

    CSV format:
    building, floor, room, ap_bssid, ap_ssid, rssi, timestamp
    """
    print(f"Učitavanje podataka iz CSV fajla: {csv_file}")

    # Učitaj CSV
    df_raw = pd.read_csv(csv_file)

    print(f"Učitano {len(df_raw)} RSSI mjerenja iz CSV fajla")

    # Grupiranje po lokaciji i kreiranje WiFi fingerprint-a
    # Za svaku lokaciju (building, floor, room), kreiramo feature vektor sa RSSI vrijednostima

    # Enkodiranje zgrada i spratova
    building_encoder = LabelEncoder()
    floor_encoder = LabelEncoder()

    df_raw['BUILDINGID'] = building_encoder.fit_transform(df_raw['building'])
    df_raw['FLOOR_ENCODED'] = floor_encoder.fit_transform(df_raw['floor'])

    # Grupe po lokaciji
    location_groups = df_raw.groupby(['building', 'floor', 'room', 'BUILDINGID', 'FLOOR_ENCODED'])

    # Kreiranje WiFi fingerprint-a
    all_aps = df_raw['ap_bssid'].unique()
    print(f"Pronađeno {len(all_aps)} jedinstvenih WiFi pristupnih tačaka")

    # Lista podataka za finalni DataFrame
    data_rows = []

    for (building, floor, room, building_id, floor_id), group in location_groups:
        # Kreiraj feature vektor za ovu lokaciju
        wifi_features = {}

        # Za svaki AP, izračunaj srednju RSSI vrijednost
        ap_rssi = group.groupby('ap_bssid')['rssi'].mean()

        for ap in all_aps:
            col_name = f'WAP_{ap.replace(":", "")}'  # Ime kolone
            wifi_features[col_name] = ap_rssi.get(ap, -100)  # -100 ako AP nije detektovan

        # Dodaj prostorne informacije
        # Za koordinate, koristimo enkodovane ID-jeve sa random offsetom
        # (pošto nemamo GPS koordinate)
        longitude = building_id * 100 + floor_id * 10 + np.random.randn() * 2
        latitude = building_id * 80 + floor_id * 8 + np.random.randn() * 2

        wifi_features['LONGITUDE'] = longitude
        wifi_features['LATITUDE'] = latitude
        wifi_features['FLOOR'] = floor_id
        wifi_features['BUILDINGID'] = building_id

        data_rows.append(wifi_features)

    # Kreiraj DataFrame
    df = pd.DataFrame(data_rows)

    # Sortiraj kolone: prvo WAP kolone, zatim ostalo
    wap_cols = sorted([col for col in df.columns if col.startswith('WAP_')])
    other_cols = ['LONGITUDE', 'LATITUDE', 'FLOOR', 'BUILDINGID']
    df = df[wap_cols + other_cols]

    print(f"Kreiran dataset sa {len(df)} lokacija i {len(wap_cols)} WiFi pristupnih tačaka")
    print(f"Zgrade: {building_encoder.classes_}")
    print(f"Spratovi: {floor_encoder.classes_}")

    return df


def generate_simulated_data():
    """
    Generisanje simuliranog dataseta (fallback ako CSV ne postoji)
    """
    print("Kreiranje simuliranog WiFi Indoor Positioning dataseta...")

    n_samples = 5000
    n_wifi_aps = 100  # Broj WiFi access point-ova

    # Simulirani WiFi signali (RSSI vrijednosti: -100 do 0 dBm)
    # -100 znači "nije detektovan"
    wifi_signals = np.random.uniform(-100, -20, (n_samples, n_wifi_aps))

    # Dodaj "nedetektovane" AP-ove (RSSI = -100)
    mask = np.random.random((n_samples, n_wifi_aps)) < 0.3
    wifi_signals[mask] = -100

    # Simulirane prostorne informacije
    buildings = np.random.randint(0, 3, n_samples)  # 3 zgrade
    floors = np.random.randint(0, 5, n_samples)     # 5 spratova

    # Koordinate zavisne od zgrade i sprata
    longitude = buildings * 100 + floors * 10 + np.random.randn(n_samples) * 5
    latitude = buildings * 80 + floors * 8 + np.random.randn(n_samples) * 4

    # Kreiranje DataFrame-a
    wifi_cols = [f'WAP{i:03d}' for i in range(n_wifi_aps)]
    df = pd.DataFrame(wifi_signals, columns=wifi_cols)
    df['LONGITUDE'] = longitude
    df['LATITUDE'] = latitude
    df['FLOOR'] = floors
    df['BUILDINGID'] = buildings

    return df


def download_and_load_data():
    """
    Učitavanje WiFi positioning dataseta.

    Prioritet:
    1. Ako postoji WIFI_DATA_FILE, učitaj podatke iz CSV-a
    2. Inače, generiši simulirane podatke
    """

    # Proveri da li postoji CSV fajl
    if os.path.exists(WIFI_DATA_FILE):
        try:
            return load_data_from_csv(WIFI_DATA_FILE)
        except Exception as e:
            print(f"Greška pri učitavanju CSV fajla: {e}")
            print("Koristiću simulirane podatke...")
            return generate_simulated_data()
    else:
        print(f"CSV fajl '{WIFI_DATA_FILE}' ne postoji.")
        print("Koristiću simulirane podatke...")
        return generate_simulated_data()

# Učitavanje podataka
df = download_and_load_data()
print(f"\nDimenzije dataseta: {df.shape}")
print(f"\nPrvih 5 redova:")
print(df.head())

# Osnovna statistika
print("\n" + "="*70)
print("EKSPLORATIVNA ANALIZA PODATAKA")
print("="*70)

print(f"\nDistribucija po zgradama:")
print(df['BUILDINGID'].value_counts().sort_index())

print(f"\nDistribucija po spratovima:")
print(df['FLOOR'].value_counts().sort_index())

# ============================================================================
# 2. PRETPROCESIRANJE PODATAKA
# ============================================================================

print("\n" + "="*70)
print("PRETPROCESIRANJE PODATAKA")
print("="*70)

# Odvajanje features i target varijabli
wifi_cols = [col for col in df.columns if col.startswith('WAP')]
X_wifi = df[wifi_cols].values

# Target varijable
y_building = df['BUILDINGID'].values
y_floor = df['FLOOR'].values
y_longitude = df['LONGITUDE'].values
y_latitude = df['LATITUDE'].values

print(f"\nBroj WiFi AP-ova: {len(wifi_cols)}")
print(f"Raspon RSSI vrijednosti: [{X_wifi.min():.2f}, {X_wifi.max():.2f}]")

# Normalizacija WiFi signala
# RSSI je u rangu -100 do 0, normalizujemo na 0-1
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_wifi)

print(f"\nNakon normalizacije:")
print(f"Mean: {X_normalized.mean():.4f}, Std: {X_normalized.std():.4f}")

# One-hot encoding za klasifikacione target-e
y_building_cat = to_categorical(y_building, num_classes=3)
y_floor_cat = to_categorical(y_floor, num_classes=5)

# ============================================================================
# 3. PODJELA NA TRAIN/VALIDATION/TEST SKUPOVE
# ============================================================================

print("\n" + "="*70)
print("PODJELA PODATAKA")
print("="*70)

# Train: 70%, Validation: 15%, Test: 15%
X_temp, X_test, y_b_temp, y_b_test, y_f_temp, y_f_test, \
    y_lon_temp, y_lon_test, y_lat_temp, y_lat_test = train_test_split(
    X_normalized, y_building_cat, y_floor_cat, y_longitude, y_latitude,
    test_size=0.15, random_state=42, stratify=y_building
)

X_train, X_val, y_b_train, y_b_val, y_f_train, y_f_val, \
    y_lon_train, y_lon_val, y_lat_train, y_lat_val = train_test_split(
    X_temp, y_b_temp, y_f_temp, y_lon_temp, y_lat_temp,
    test_size=0.176, random_state=42  # 0.176 * 0.85 ≈ 0.15
)

print(f"Train set: {X_train.shape[0]} uzoraka")
print(f"Validation set: {X_val.shape[0]} uzoraka")
print(f"Test set: {X_test.shape[0]} uzoraka")

# ============================================================================
# 4. ARHITEKTURA MODELA
# ============================================================================

print("\n" + "="*70)
print("KREIRANJE MODELA")
print("="*70)

def create_multi_output_model(input_dim, learning_rate=0.001, dropout_rate=0.3):
    """
    Multi-output model sa tri izlaza:
    1. Klasifikacija zgrade (3 klase)
    2. Klasifikacija sprata (5 klasa)
    3. Regresija koordinata (longitude, latitude)
    """

    # Input layer
    inputs = layers.Input(shape=(input_dim,), name='wifi_input')

    # Shared dense layers
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate/2)(x)

    # Output 1: Building classification
    building_branch = layers.Dense(32, activation='relu', name='building_dense')(x)
    building_output = layers.Dense(3, activation='softmax', name='building_output')(building_branch)

    # Output 2: Floor classification
    floor_branch = layers.Dense(32, activation='relu', name='floor_dense')(x)
    floor_output = layers.Dense(5, activation='softmax', name='floor_output')(floor_branch)

    # Output 3: Coordinate regression
    coord_branch = layers.Dense(32, activation='relu', name='coord_dense')(x)
    coord_output = layers.Dense(2, activation='linear', name='coord_output')(coord_branch)

    # Kreiranje modela
    model = models.Model(
        inputs=inputs,
        outputs=[building_output, floor_output, coord_output]
    )

    # Kompilacija
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'building_output': 'categorical_crossentropy',
            'floor_output': 'categorical_crossentropy',
            'coord_output': 'mse'
        },
        loss_weights={
            'building_output': 1.0,
            'floor_output': 1.0,
            'coord_output': 0.1  # Manja težina za regresiju
        },
        metrics={
            'building_output': ['accuracy'],
            'floor_output': ['accuracy'],
            'coord_output': ['mae']
        }
    )

    return model

# Kreiranje modela
model = create_multi_output_model(
    input_dim=X_train.shape[1],
    learning_rate=0.001,
    dropout_rate=0.3
)

print("\nArhitektura modela:")
model.summary()

# ============================================================================
# 5. TRENIRANJE MODELA
# ============================================================================

print("\n" + "="*70)
print("TRENIRANJE MODELA")
print("="*70)

# Priprema target varijabli za treniranje
y_train = {
    'building_output': y_b_train,
    'floor_output': y_f_train,
    'coord_output': np.column_stack([y_lon_train, y_lat_train])
}

y_val = {
    'building_output': y_b_val,
    'floor_output': y_f_val,
    'coord_output': np.column_stack([y_lon_val, y_lat_val])
}

# Callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-6,
    verbose=1
)

# Treniranje
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# ============================================================================
# 6. EVALUACIJA NA TEST SKUPU
# ============================================================================

print("\n" + "="*70)
print("EVALUACIJA NA TEST SKUPU")
print("="*70)

# Priprema test podataka
y_test = {
    'building_output': y_b_test,
    'floor_output': y_f_test,
    'coord_output': np.column_stack([y_lon_test, y_lat_test])
}

# Predikcija
predictions = model.predict(X_test)
y_pred_building = predictions[0]
y_pred_floor = predictions[1]
y_pred_coords = predictions[2]

# Evaluacija klasifikacije zgrade
building_pred_classes = np.argmax(y_pred_building, axis=1)
building_true_classes = np.argmax(y_b_test, axis=1)
building_accuracy = accuracy_score(building_true_classes, building_pred_classes)

print(f"\n--- KLASIFIKACIJA ZGRADE ---")
print(f"Tačnost: {building_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(building_true_classes, building_pred_classes,
                          target_names=[f'Building {i}' for i in range(3)]))

# Evaluacija klasifikacije sprata
floor_pred_classes = np.argmax(y_pred_floor, axis=1)
floor_true_classes = np.argmax(y_f_test, axis=1)
floor_accuracy = accuracy_score(floor_true_classes, floor_pred_classes)

print(f"\n--- KLASIFIKACIJA SPRATA ---")
print(f"Tačnost: {floor_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(floor_true_classes, floor_pred_classes,
                          target_names=[f'Floor {i}' for i in range(5)]))

# Evaluacija regresije koordinata
coord_mae = mean_absolute_error(y_pred_coords,
                                np.column_stack([y_lon_test, y_lat_test]))
coord_rmse = np.sqrt(mean_squared_error(y_pred_coords,
                                        np.column_stack([y_lon_test, y_lat_test])))

print(f"\n--- REGRESIJA KOORDINATA ---")
print(f"Mean Absolute Error: {coord_mae:.4f} m")
print(f"Root Mean Square Error: {coord_rmse:.4f} m")

# ============================================================================
# 7. VIZUALIZACIJA REZULTATA
# ============================================================================

print("\n" + "="*70)
print("VIZUALIZACIJA")
print("="*70)

# Plotting setup
fig = plt.figure(figsize=(18, 12))

# 1. Training history - Loss
ax1 = plt.subplot(3, 3, 1)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Total Loss Tokom Treniranja')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Building accuracy
ax2 = plt.subplot(3, 3, 2)
plt.plot(history.history['building_output_accuracy'], label='Train', linewidth=2)
plt.plot(history.history['val_building_output_accuracy'], label='Val', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Building Classification Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Floor accuracy
ax3 = plt.subplot(3, 3, 3)
plt.plot(history.history['floor_output_accuracy'], label='Train', linewidth=2)
plt.plot(history.history['val_floor_output_accuracy'], label='Val', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Floor Classification Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Confusion matrix - Building
ax4 = plt.subplot(3, 3, 4)
cm_building = confusion_matrix(building_true_classes, building_pred_classes)
sns.heatmap(cm_building, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'B{i}' for i in range(3)],
            yticklabels=[f'B{i}' for i in range(3)])
plt.title('Confusion Matrix - Zgrada')
plt.ylabel('True')
plt.xlabel('Predicted')

# 5. Confusion matrix - Floor
ax5 = plt.subplot(3, 3, 5)
cm_floor = confusion_matrix(floor_true_classes, floor_pred_classes)
sns.heatmap(cm_floor, annot=True, fmt='d', cmap='Greens',
            xticklabels=[f'F{i}' for i in range(5)],
            yticklabels=[f'F{i}' for i in range(5)])
plt.title('Confusion Matrix - Sprat')
plt.ylabel('True')
plt.xlabel('Predicted')

# 6. Coordinate predictions - Longitude
ax6 = plt.subplot(3, 3, 6)
plt.scatter(y_lon_test, y_pred_coords[:, 0], alpha=0.5, s=10)
plt.plot([y_lon_test.min(), y_lon_test.max()],
         [y_lon_test.min(), y_lon_test.max()], 'r--', linewidth=2)
plt.xlabel('True Longitude')
plt.ylabel('Predicted Longitude')
plt.title('Longitude Predictions')
plt.grid(True, alpha=0.3)

# 7. Coordinate predictions - Latitude
ax7 = plt.subplot(3, 3, 7)
plt.scatter(y_lat_test, y_pred_coords[:, 1], alpha=0.5, s=10)
plt.plot([y_lat_test.min(), y_lat_test.max()],
         [y_lat_test.min(), y_lat_test.max()], 'r--', linewidth=2)
plt.xlabel('True Latitude')
plt.ylabel('Predicted Latitude')
plt.title('Latitude Predictions')
plt.grid(True, alpha=0.3)

# 8. 2D Position visualization
ax8 = plt.subplot(3, 3, 8)
# Prikaži samo sample za čitljivost
sample_idx = np.random.choice(len(y_lon_test), 200, replace=False)
plt.scatter(y_lon_test[sample_idx], y_lat_test[sample_idx],
           c='blue', label='True', alpha=0.6, s=30)
plt.scatter(y_pred_coords[sample_idx, 0], y_pred_coords[sample_idx, 1],
           c='red', label='Predicted', alpha=0.6, s=30, marker='x')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('2D Pozicije (Sample)')
plt.legend()
plt.grid(True, alpha=0.3)

# 9. Positioning error distribution
ax9 = plt.subplot(3, 3, 9)
errors = np.sqrt((y_lon_test - y_pred_coords[:, 0])**2 +
                 (y_lat_test - y_pred_coords[:, 1])**2)
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Greška Pozicioniranja (m)')
plt.ylabel('Frekvencija')
plt.title(f'Distribucija Greške\n(Median: {np.median(errors):.2f}m)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('wifi_positioning_results.png', dpi=150, bbox_inches='tight')
print("\nGrafici sačuvani kao 'wifi_positioning_results.png'")
plt.show()


# ============================================================================
# 8. HYPERPARAMETER TUNING (Bonus)
# ============================================================================

print("\n" + "="*70)
print("HYPERPARAMETER TUNING")
print("="*70)

def tune_hyperparameters():
    """
    Jednostavno grid search preko ključnih hiperparametara
    """

    learning_rates = [0.001, 0.0005]
    dropout_rates = [0.2, 0.3, 0.4]

    results = []

    for lr in learning_rates:
        for dropout in dropout_rates:
            print(f"\nTestiranje: LR={lr}, Dropout={dropout}")

            # Kreiraj model
            tuning_model = create_multi_output_model(
                input_dim=X_train.shape[1],
                learning_rate=lr,
                dropout_rate=dropout
            )

            # Kraće treniranje za tuning
            tuning_history = tuning_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=30,
                batch_size=64,
                callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
                verbose=0
            )

            # Evaluacija
            val_loss = min(tuning_history.history['val_loss'])

            results.append({
                'lr': lr,
                'dropout': dropout,
                'val_loss': val_loss
            })

            print(f"  Val Loss: {val_loss:.4f}")

    # Najbolji rezultati
    best = min(results, key=lambda x: x['val_loss'])
    print(f"\n{'='*50}")
    print(f"NAJBOLJI HIPERPARAMETRI:")
    print(f"  Learning Rate: {best['lr']}")
    print(f"  Dropout Rate: {best['dropout']}")
    print(f"  Validation Loss: {best['val_loss']:.4f}")
    print(f"{'='*50}")

    return results

# Pokreni tuning (opcionalno - može trajati duže)
tuning_results = tune_hyperparameters()

print("\n" + "="*70)
print("DEMO ZAVRŠEN!")
print("="*70)
print("\nŠta smo demonstrirali:")
print("✓ Učitavanje i eksploraciju WiFi positioning dataseta")
print("✓ Pretprocesiranje (normalizacija, encoding)")
print("✓ Podjelu na Train/Val/Test")
print("✓ Multi-output arhitekturu (klasifikacija + regresija)")
print("✓ Treniranje sa callbacks (Early Stopping, LR Reduction)")
print("✓ Evaluaciju performansi (accuracy, MAE, RMSE)")
print("✓ Vizualizaciju rezultata")
print("✓ Osnove hyperparameter tuninga")
