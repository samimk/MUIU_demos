"""
WiFi Indoor Positioning System - Kompletna Implementacija
=========================================================
Dataset: WiFi fingerprinting za pozicioniranje u zgradi
Cilj: Predviđanje pozicije (zgrada, sprat, prostorija) na osnovu WiFi signala
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
import pickle
warnings.filterwarnings('ignore')

# Postavljanje seed-a za reproducibilnost
np.random.seed(42)
keras.utils.set_random_seed(42)

# Putanja do CSV fajla sa WiFi podacima
# Ako fajl ne postoji, koristiće se simulirani podaci
WIFI_DATA_FILE = "wifi_data_20251117_171358.csv"

# Ime fajla za pohranu istrenirane mreže
TRAINED_ANN_FILE = "wifi_localization_model.keras"

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

    # Enkodiranje zgrada, spratova i prostorija
    building_encoder = LabelEncoder()
    floor_encoder = LabelEncoder()
    room_encoder = LabelEncoder()

    df_raw['BUILDINGID'] = building_encoder.fit_transform(df_raw['building'])
    df_raw['FLOOR_ENCODED'] = floor_encoder.fit_transform(df_raw['floor'])
    df_raw['ROOM_ENCODED'] = room_encoder.fit_transform(df_raw['room'])

    # Grupe po lokaciji
    location_groups = df_raw.groupby(['building', 'floor', 'room', 'BUILDINGID', 'FLOOR_ENCODED', 'ROOM_ENCODED'])

    # Kreiranje WiFi fingerprint-a
    all_aps = df_raw['ap_bssid'].unique()
    print(f"Pronađeno {len(all_aps)} jedinstvenih WiFi pristupnih tačaka")

    # Lista podataka za finalni DataFrame
    data_rows = []

    for (building, floor, room, building_id, floor_id, room_id), group in location_groups:
        # Kreiraj feature vektor za ovu lokaciju
        wifi_features = {}

        # Za svaki AP, izračunaj srednju RSSI vrijednost
        ap_rssi = group.groupby('ap_bssid')['rssi'].mean()

        for ap in all_aps:
            col_name = f'WAP_{ap.replace(":", "")}'  # Ime kolone
            wifi_features[col_name] = ap_rssi.get(ap, -100)  # -100 ako AP nije detektovan

        # Dodaj lokacijske informacije
        wifi_features['BUILDINGID'] = building_id
        wifi_features['FLOOR'] = floor_id
        wifi_features['ROOMID'] = room_id

        data_rows.append(wifi_features)

    # Kreiraj DataFrame
    df = pd.DataFrame(data_rows)

    # Sortiraj kolone: prvo WAP kolone, zatim ostalo
    wap_cols = sorted([col for col in df.columns if col.startswith('WAP_')])
    other_cols = ['BUILDINGID', 'FLOOR', 'ROOMID']
    df = df[wap_cols + other_cols]

    print(f"Kreiran dataset sa {len(df)} lokacija i {len(wap_cols)} WiFi pristupnih tačaka")
    print(f"Zgrade: {building_encoder.classes_}")
    print(f"Spratovi: {floor_encoder.classes_}")
    print(f"Prostorije: {room_encoder.classes_}")

    # Spremanje encoder-a za kasnije dekodiranje
    encoders = {
        'building': building_encoder,
        'floor': floor_encoder,
        'room': room_encoder
    }

    return df, encoders


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

    # Simulirane lokacijske informacije
    buildings = np.random.randint(0, 3, n_samples)  # 3 zgrade
    floors = np.random.randint(0, 5, n_samples)     # 5 spratova
    rooms = np.random.randint(0, 10, n_samples)     # 10 prostorija

    # Kreiranje DataFrame-a
    wifi_cols = [f'WAP{i:03d}' for i in range(n_wifi_aps)]
    df = pd.DataFrame(wifi_signals, columns=wifi_cols)
    df['BUILDINGID'] = buildings
    df['FLOOR'] = floors
    df['ROOMID'] = rooms

    # Kreiranje dummy encoder-a za simulirane podatke
    building_encoder = LabelEncoder()
    floor_encoder = LabelEncoder()
    room_encoder = LabelEncoder()

    building_encoder.fit(['Building 0', 'Building 1', 'Building 2'])
    floor_encoder.fit(['Floor 0', 'Floor 1', 'Floor 2', 'Floor 3', 'Floor 4'])
    room_encoder.fit([f'Room {i}' for i in range(10)])

    encoders = {
        'building': building_encoder,
        'floor': floor_encoder,
        'room': room_encoder
    }

    return df, encoders


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
df, encoders = download_and_load_data()
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
y_room = df['ROOMID'].values

print(f"\nBroj WiFi AP-ova: {len(wifi_cols)}")
print(f"Raspon RSSI vrijednosti: [{X_wifi.min():.2f}, {X_wifi.max():.2f}]")

# Dinamički odrediti broj klasa
num_buildings = len(np.unique(y_building))
num_floors = len(np.unique(y_floor))
num_rooms = len(np.unique(y_room))
print(f"\nBroj jedinstvenih zgrada: {num_buildings}")
print(f"Broj jedinstvenih spratova: {num_floors}")
print(f"Broj jedinstvenih prostorija: {num_rooms}")

# Normalizacija WiFi signala
# RSSI je u rangu -100 do 0, normalizujemo na 0-1
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_wifi)

print(f"\nNakon normalizacije:")
print(f"Mean: {X_normalized.mean():.4f}, Std: {X_normalized.std():.4f}")

# One-hot encoding za klasifikacione target-e
y_building_cat = to_categorical(y_building, num_classes=num_buildings)
y_floor_cat = to_categorical(y_floor, num_classes=num_floors)
y_room_cat = to_categorical(y_room, num_classes=num_rooms)

# ============================================================================
# 3. PODJELA NA TRAIN/VALIDATION/TEST SKUPOVE
# ============================================================================

print("\n" + "="*70)
print("PODJELA PODATAKA")
print("="*70)

# Train: 70%, Validation: 15%, Test: 15%
X_temp, X_test, y_b_temp, y_b_test, y_f_temp, y_f_test, \
    y_r_temp, y_r_test = train_test_split(
    X_normalized, y_building_cat, y_floor_cat, y_room_cat,
    test_size=0.15, random_state=42, stratify=y_building
)

X_train, X_val, y_b_train, y_b_val, y_f_train, y_f_val, \
    y_r_train, y_r_val = train_test_split(
    X_temp, y_b_temp, y_f_temp, y_r_temp,
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

def create_multi_output_model(input_dim, num_buildings, num_floors, num_rooms, learning_rate=0.001, dropout_rate=0.3):
    """
    Multi-output model sa tri izlaza:
    1. Klasifikacija zgrade (dinamički broj klasa)
    2. Klasifikacija sprata (dinamički broj klasa)
    3. Klasifikacija prostorije (dinamički broj klasa)
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
    building_output = layers.Dense(num_buildings, activation='softmax', name='building_output')(building_branch)

    # Output 2: Floor classification
    floor_branch = layers.Dense(32, activation='relu', name='floor_dense')(x)
    floor_output = layers.Dense(num_floors, activation='softmax', name='floor_output')(floor_branch)

    # Output 3: Room classification
    room_branch = layers.Dense(32, activation='relu', name='room_dense')(x)
    room_output = layers.Dense(num_rooms, activation='softmax', name='room_output')(room_branch)

    # Kreiranje modela
    model = models.Model(
        inputs=inputs,
        outputs=[building_output, floor_output, room_output]
    )

    # Kompilacija
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'building_output': 'categorical_crossentropy',
            'floor_output': 'categorical_crossentropy',
            'room_output': 'categorical_crossentropy'
        },
        loss_weights={
            'building_output': 1.0,
            'floor_output': 1.0,
            'room_output': 1.0
        },
        metrics={
            'building_output': ['accuracy'],
            'floor_output': ['accuracy'],
            'room_output': ['accuracy']
        }
    )

    return model

# Kreiranje modela
model = create_multi_output_model(
    input_dim=X_train.shape[1],
    num_buildings=num_buildings,
    num_floors=num_floors,
    num_rooms=num_rooms,
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
    'room_output': y_r_train
}

y_val = {
    'building_output': y_b_val,
    'floor_output': y_f_val,
    'room_output': y_r_val
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
# 5.1. POHRANA ISTRENIRANE MREŽE
# ============================================================================

print("\n" + "="*70)
print("POHRANA MODELA")
print("="*70)

# Pohrana modela
model.save(TRAINED_ANN_FILE)
print(f"Model sačuvan u: {TRAINED_ANN_FILE}")

# Pohrana scaler-a i dodatnih informacija potrebnih za upotrebu modela
model_metadata = {
    'scaler': scaler,
    'num_buildings': num_buildings,
    'num_floors': num_floors,
    'num_rooms': num_rooms,
    'wifi_cols': wifi_cols,
    'encoders': encoders
}

metadata_file = TRAINED_ANN_FILE.replace('.keras', '_metadata.pkl')
with open(metadata_file, 'wb') as f:
    pickle.dump(model_metadata, f)
print(f"Metadata sačuvana u: {metadata_file}")

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
    'room_output': y_r_test
}

# Predikcija
predictions = model.predict(X_test)
y_pred_building = predictions[0]
y_pred_floor = predictions[1]
y_pred_room = predictions[2]

# Evaluacija klasifikacije zgrade
building_pred_classes = np.argmax(y_pred_building, axis=1)
building_true_classes = np.argmax(y_b_test, axis=1)
building_accuracy = accuracy_score(building_true_classes, building_pred_classes)

print(f"\n--- KLASIFIKACIJA ZGRADE ---")
print(f"Tačnost: {building_accuracy:.4f}")
print("\nClassification Report:")
# Generiši target_names samo za klase koje postoje u test podacima
unique_building_classes = np.unique(np.concatenate([building_true_classes, building_pred_classes]))
building_target_names = [f'Building {i}' for i in unique_building_classes]
print(classification_report(building_true_classes, building_pred_classes,
                          target_names=building_target_names, labels=unique_building_classes))

# Evaluacija klasifikacije sprata
floor_pred_classes = np.argmax(y_pred_floor, axis=1)
floor_true_classes = np.argmax(y_f_test, axis=1)
floor_accuracy = accuracy_score(floor_true_classes, floor_pred_classes)

print(f"\n--- KLASIFIKACIJA SPRATA ---")
print(f"Tačnost: {floor_accuracy:.4f}")
print("\nClassification Report:")
# Generiši target_names samo za klase koje postoje u test podacima
unique_floor_classes = np.unique(np.concatenate([floor_true_classes, floor_pred_classes]))
floor_target_names = [f'Floor {i}' for i in unique_floor_classes]
print(classification_report(floor_true_classes, floor_pred_classes,
                          target_names=floor_target_names, labels=unique_floor_classes))

# Evaluacija klasifikacije prostorije
room_pred_classes = np.argmax(y_pred_room, axis=1)
room_true_classes = np.argmax(y_r_test, axis=1)
room_accuracy = accuracy_score(room_true_classes, room_pred_classes)

print(f"\n--- KLASIFIKACIJA PROSTORIJE ---")
print(f"Tačnost: {room_accuracy:.4f}")
print("\nClassification Report:")
# Generiši target_names samo za klase koje postoje u test podacima
unique_room_classes = np.unique(np.concatenate([room_true_classes, room_pred_classes]))
room_target_names = [f'Room {i}' for i in unique_room_classes]
print(classification_report(room_true_classes, room_pred_classes,
                          target_names=room_target_names, labels=unique_room_classes))

# ============================================================================
# 7. VIZUALIZACIJA REZULTATA
# ============================================================================

print("\n" + "="*70)
print("VIZUALIZACIJA")
print("="*70)

# Plotting setup
fig = plt.figure(figsize=(18, 10))

# 1. Training history - Loss
ax1 = plt.subplot(2, 3, 1)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Total Loss Tokom Treniranja')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Building accuracy
ax2 = plt.subplot(2, 3, 2)
plt.plot(history.history['building_output_accuracy'], label='Train', linewidth=2)
plt.plot(history.history['val_building_output_accuracy'], label='Val', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Building Classification Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Floor accuracy
ax3 = plt.subplot(2, 3, 3)
plt.plot(history.history['floor_output_accuracy'], label='Train', linewidth=2)
plt.plot(history.history['val_floor_output_accuracy'], label='Val', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Floor Classification Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Room accuracy
ax4 = plt.subplot(2, 3, 4)
plt.plot(history.history['room_output_accuracy'], label='Train', linewidth=2)
plt.plot(history.history['val_room_output_accuracy'], label='Val', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Room Classification Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# 5. Confusion matrix - Building
ax5 = plt.subplot(2, 3, 5)
cm_building = confusion_matrix(building_true_classes, building_pred_classes, labels=unique_building_classes)
sns.heatmap(cm_building, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'B{i}' for i in unique_building_classes],
            yticklabels=[f'B{i}' for i in unique_building_classes])
plt.title('Confusion Matrix - Zgrada')
plt.ylabel('True')
plt.xlabel('Predicted')

# 6. Confusion matrix - Floor
ax6 = plt.subplot(2, 3, 6)
cm_floor = confusion_matrix(floor_true_classes, floor_pred_classes, labels=unique_floor_classes)
# Limit broj tickova za čitljivost
max_ticks = 20
if len(unique_floor_classes) > max_ticks:
    tick_step = len(unique_floor_classes) // max_ticks
    tick_labels_x = [f'F{i}' if idx % tick_step == 0 else '' for idx, i in enumerate(unique_floor_classes)]
    tick_labels_y = [f'F{i}' if idx % tick_step == 0 else '' for idx, i in enumerate(unique_floor_classes)]
else:
    tick_labels_x = [f'F{i}' for i in unique_floor_classes]
    tick_labels_y = [f'F{i}' for i in unique_floor_classes]

sns.heatmap(cm_floor, annot=False, cmap='Greens',
            xticklabels=tick_labels_x,
            yticklabels=tick_labels_y)
plt.title('Confusion Matrix - Sprat')
plt.ylabel('True')
plt.xlabel('Predicted')

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
                num_buildings=num_buildings,
                num_floors=num_floors,
                num_rooms=num_rooms,
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
#tuning_results = tune_hyperparameters()

print("\n" + "="*70)
print("DEMO ZAVRŠEN!")
print("="*70)
print("\nŠta smo demonstrirali:")
print("✓ Učitavanje i eksploraciju WiFi positioning dataseta")
print("✓ Pretprocesiranje (normalizacija, encoding)")
print("✓ Podjelu na Train/Val/Test")
print("✓ Multi-output arhitekturu (3 klasifikaciona izlaza)")
print("✓ Treniranje sa callbacks (Early Stopping, LR Reduction)")
print("✓ Evaluaciju performansi (accuracy za zgrada/sprat/prostorija)")
print("✓ Vizualizaciju rezultata")
print("✓ Osnove hyperparameter tuninga")
print("\nLokacija se predviđa kroz 3 dimenzije:")
print("  - Zgrada")
print("  - Sprat")
print("  - Prostorija")
