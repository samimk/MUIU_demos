# WiFi Indoor Localization System

Sistem za prikupljanje WiFi podataka i treniranje neuronske mreže za određivanje lokacije u zgradi na osnovu WiFi signala.

## Pregled

Ovaj sistem se sastoji od dvije aplikacije:

1. **get_wifi_data.py** - GUI aplikacija za prikupljanje WiFi RSSI podataka sa anotacijom lokacije
2. **wifi_localization_demo.py** - Demo aplikacija koja trenira multi-output neuronsku mrežu za predviđanje lokacije

## Instalacija

Potrebne biblioteke:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn keras tensorflow tkinter
```

## 1. Prikupljanje podataka (get_wifi_data.py)

### Karakteristike

- **GUI interfejs** baziran na Tkinter-u
- **Periodično skeniranje** WiFi mreža (podesivi interval)
- **Anotacija lokacije** kroz combo box-ove (zgrada, sprat, prostorija)
- **Konfigurabilna struktura** zgrada/spratova/prostorija kroz JSON fajl
- **Izvoz podataka** u CSV format

### Konfiguracija lokacija

Zgrade, spratovi i prostorije se konfigurišu u `wifi_config.json` fajlu:

```json
{
  "buildings": [
    {
      "name": "Zgrada A",
      "floors": [
        {
          "name": "Prizemlje",
          "rooms": ["Ulaz", "Recepcija", "Sala 101", "WC"]
        },
        {
          "name": "Sprat 1",
          "rooms": ["Kancelarija 201", "Laboratorija 1", "WC"]
        }
      ]
    }
  ]
}
```

### Korišćenje

1. Pokrenite aplikaciju:
   ```bash
   python get_wifi_data.py
   ```

2. Odaberite lokaciju (zgrada, sprat, prostorija)

3. Podesite period mjerenja (default: 5 sekundi)

4. Kliknite **"Započni prikupljanje"** za početak skeniranja WiFi mreža

5. Aplikacija će periodično:
   - Skenirati dostupne WiFi pristupne tačke
   - Očitati RSSI vrijednosti
   - Dodati podatke u tabelu

6. Kontrole:
   - **Zaustavi prikupljanje** - privremeno zaustavlja mjerenje
   - **Resetiraj prikupljanje** - briše sve prikupljene podatke
   - **Izvezi podatke** - snima podatke u CSV fajl

### Format CSV fajla

Izvezeni CSV fajl sadrži sljedeća polja:

| Polje      | Opis                                    |
|------------|-----------------------------------------|
| building   | Naziv zgrade                            |
| floor      | Naziv sprata                            |
| room       | Naziv prostorije                        |
| ap_bssid   | MAC adresa WiFi pristupne tačke (AP)    |
| ap_ssid    | Naziv WiFi mreže                        |
| rssi       | Jačina signala u dBm (-100 do 0)        |
| timestamp  | Vrijeme mjerenja (HH:MM:SS)             |

### WiFi Skeniranje

Aplikacija pokušava da koristi sljedeće metode za skeniranje:

1. **nmcli** (NetworkManager CLI) - preporučeno za Linux sisteme
2. **iwlist** - alternativa (zahtijeva sudo privilegije)
3. **Simulirani podaci** - fallback za testiranje bez WiFi

## 2. Treniranje modela (wifi_localization_demo.py)

### Karakteristike

- **Multi-output neuronska mreža** sa tri izlaza:
  - Klasifikacija zgrade
  - Klasifikacija sprata
  - Regresija koordinata (longitude, latitude)

- **Automatsko učitavanje podataka**:
  - Učitava prave podatke iz CSV fajla (ako postoji)
  - Koristi simulirane podatke kao fallback

- **Kompletna evaluacija**:
  - Confusion matrices
  - Classification reports
  - Positioning error distribucija
  - Training curves

### Konfiguracija

Na početku `wifi_localization_demo.py` fajla:

```python
# Putanja do CSV fajla sa WiFi podacima
WIFI_DATA_FILE = "wifi_collected_data.csv"
```

Postavite ime CSV fajla koji ste izvezli iz `get_wifi_data.py` aplikacije.

### Korišćenje

1. Prikupite podatke koristeći `get_wifi_data.py`
2. Izvezite podatke u CSV fajl (npr. `wifi_collected_data.csv`)
3. Postavite `WIFI_DATA_FILE` varijablu na ime vašeg CSV fajla
4. Pokrenite demo:
   ```bash
   python wifi_localization_demo.py
   ```

### Arhitektura modela

```
Input (WiFi RSSI signali)
    ↓
Dense(256) + BatchNorm + Dropout
    ↓
Dense(128) + BatchNorm + Dropout
    ↓
Dense(64) + Dropout
    ↓
    ├─→ Building Classification (3 klase)
    ├─→ Floor Classification (5 klasa)
    └─→ Coordinate Regression (longitude, latitude)
```

### Izlazi

- **Console output**: Detaljna statistika i evaluacija
- **Grafici**: `wifi_positioning_results.png` sa 9 subplots-a
  - Training loss
  - Accuracy curves
  - Confusion matrices
  - Coordinate predictions
  - Error distribution

## Workflow

1. **Priprema**:
   - Konfiguriši `wifi_config.json` sa strukturom vaših zgrada

2. **Prikupljanje podataka**:
   - Pokrenuti `get_wifi_data.py`
   - Obiđi različite lokacije
   - Za svaku lokaciju prikupi mjerenja (preporučeno: bar 20-30 mjerenja)
   - Izvezi podatke u CSV

3. **Treniranje**:
   - Postavi `WIFI_DATA_FILE` u `wifi_localization_demo.py`
   - Pokreni demo
   - Analiziraj rezultate

4. **Iteracija**:
   - Prikupi više podataka za slabo klasifikovane lokacije
   - Eksperimentiši sa hiperparametrima
   - Evaluiraj performanse

## Napomene

### Kvalitet podataka

Za dobre rezultate potrebno je:

- **Dovoljno mjerenja** po lokaciji (bar 20-30)
- **Različite pozicije** u prostoriji
- **Konzistentne okolnosti** (isti deo dana, slične okolnosti)
- **Pokrivanje svih prostorija** koje želite da lokalizujete

### Limitacije

- Potreban je WiFi adapter koji podržava scanning mode
- RSSI vrijednosti variraju zavisno od:
  - Prepreka (zidovi, nameštaj)
  - Broja korisnika na mreži
  - Vremena dana
  - Pozicije uređaja

### Optimizacija

Za bolje rezultate:

1. Prikupite više podataka
2. Koristite različite pozicije u prostoriji
3. Normalizujte podatke po lokaciji
4. Eksperimentišite sa dubokim mrežama
5. Koristite ansambli modela

## Troubleshooting

### WiFi skeniranje ne radi

- Provjerite da li je WiFi adapter uključen
- Pokušajte sa `sudo` privilegijama
- Koristite simulirane podatke za testiranje

### Mala tačnost modela

- Prikupite više podataka
- Osigurajte balans između klasa (približno jednak broj uzoraka po lokaciji)
- Povećajte broj epoha treniranja
- Eksperimentišite sa arhitekturom mreže

### CSV učitavanje ne radi

- Provjerite format CSV fajla
- Osigurajte da postoje sva obavezna polja
- Provjerite encoding (UTF-8)

## Autor

Kreiran za demonstraciju WiFi indoor positioning sistema sa deep learning pristupom.
