"""
WiFi Indoor Localization GUI
=============================
Aplikacija za WiFi lokalizaciju korištenjem istrenirane ANN mreže
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import keras
import pickle
import subprocess
import re
import threading
import time
from datetime import datetime


class WiFiLocalizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("WiFi Indoor Localization")
        self.root.geometry("800x600")

        # Kreiranje menija
        self.create_menu()

        # Model i metadata
        self.model = None
        self.scaler = None
        self.num_buildings = None
        self.num_floors = None
        self.num_rooms = None
        self.wifi_cols = None
        self.encoders = None
        self.model_loaded = False

        # Lokalizacija
        self.is_localizing = False
        self.localization_thread = None
        self.localization_period = 3  # sekundi

        # Kreiranje GUI-a
        self.create_gui()

    def create_menu(self):
        """Kreiranje menija"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Help meni
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Uputstva", command=self.show_instructions)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)

    def show_about(self):
        """Prikaz About dijaloga"""
        messagebox.showinfo("About",
                           "Mašinsko učenje i inteligentno upravljanje\n\n"
                           "Red. prof. dr Samim Konjicija\n\n"
                           "Novembar 2025. godine")

    def show_instructions(self):
        """Prikaz uputstava za korištenje"""
        instructions = """UPUTSTVA ZA KORIŠTENJE

1. UČITAVANJE MODELA:
   - Kliknite na dugme "Učitaj mrežu"
   - Odaberite .keras fajl sa istreniranim modelom
   - Model mora biti treniran korištenjem wifi_localization_demo.py

2. POKRETANJE LOKALIZACIJE:
   - Nakon učitavanja modela, kliknite "Lokalizacija"
   - Aplikacija će periodično skenirati WiFi mreže
   - Trenutna lokacija će se prikazivati u realnom vremenu

3. ZAUSTAVLJANJE LOKALIZACIJE:
   - Kliknite "Zaustavi lokalizaciju" za prekid skeniranja

4. REZULTATI:
   - Zgrada: Predviđena zgrada
   - Sprat: Predviđeni sprat
   - Prostorija: Predviđena prostorija
   - Tačnost: Vjerovatnoća predvikcije

NAPOMENA: Aplikacija koristi nmcli komandu za skeniranje
WiFi mreža. Ako komanda nije dostupna, koristit će se
simulirani podaci za testiranje."""

        # Kreiraj novi prozor za uputstva
        instructions_window = tk.Toplevel(self.root)
        instructions_window.title("Uputstva za korištenje")
        instructions_window.geometry("600x500")

        # Text widget sa scrollbar-om
        text_frame = ttk.Frame(instructions_window, padding=10)
        text_frame.pack(fill="both", expand=True)

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")

        text_widget = tk.Text(text_frame, wrap="word", yscrollcommand=scrollbar.set,
                             font=("Courier", 10))
        text_widget.pack(fill="both", expand=True)
        scrollbar.config(command=text_widget.yview)

        text_widget.insert("1.0", instructions)
        text_widget.config(state="disabled")

        # Dugme za zatvaranje
        close_button = ttk.Button(instructions_window, text="Zatvori",
                                  command=instructions_window.destroy)
        close_button.pack(pady=10)

    def create_gui(self):
        """Kreiranje GUI elemenata"""

        # === Sekcija za učitavanje modela ===
        model_frame = ttk.LabelFrame(self.root, text="Model", padding=10)
        model_frame.pack(fill="x", padx=10, pady=5)

        self.load_button = ttk.Button(model_frame, text="Učitaj mrežu",
                                      command=self.load_model)
        self.load_button.pack(side="left", padx=5)

        self.model_label = ttk.Label(model_frame, text="Model nije učitan",
                                     foreground="red")
        self.model_label.pack(side="left", padx=10)

        # === Kontrole za lokalizaciju ===
        control_frame = ttk.LabelFrame(self.root, text="Kontrole", padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)

        self.start_button = ttk.Button(control_frame, text="Lokalizacija",
                                       command=self.start_localization,
                                       state="disabled")
        self.start_button.pack(side="left", padx=5)

        self.stop_button = ttk.Button(control_frame, text="Zaustavi lokalizaciju",
                                      command=self.stop_localization,
                                      state="disabled")
        self.stop_button.pack(side="left", padx=5)

        # Podešavanje perioda
        ttk.Label(control_frame, text="Period (sekundi):").pack(side="left", padx=10)
        self.period_var = tk.StringVar(value="3")
        period_spinbox = ttk.Spinbox(control_frame, from_=1, to=30,
                                    textvariable=self.period_var, width=5)
        period_spinbox.pack(side="left", padx=5)

        # === Rezultati lokalizacije ===
        results_frame = ttk.LabelFrame(self.root, text="Trenutna Lokacija", padding=10)
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Grid za rezultate
        result_grid = ttk.Frame(results_frame)
        result_grid.pack(fill="both", expand=True, pady=10)

        # Zgrada
        ttk.Label(result_grid, text="Zgrada:", font=("Arial", 12, "bold")).grid(
            row=0, column=0, sticky="w", padx=10, pady=5)
        self.building_result = ttk.Label(result_grid, text="N/A",
                                        font=("Arial", 12), foreground="blue")
        self.building_result.grid(row=0, column=1, sticky="w", padx=10, pady=5)

        ttk.Label(result_grid, text="Tačnost:", font=("Arial", 10)).grid(
            row=0, column=2, sticky="w", padx=10, pady=5)
        self.building_conf = ttk.Label(result_grid, text="N/A", font=("Arial", 10))
        self.building_conf.grid(row=0, column=3, sticky="w", padx=10, pady=5)

        # Sprat
        ttk.Label(result_grid, text="Sprat:", font=("Arial", 12, "bold")).grid(
            row=1, column=0, sticky="w", padx=10, pady=5)
        self.floor_result = ttk.Label(result_grid, text="N/A",
                                      font=("Arial", 12), foreground="green")
        self.floor_result.grid(row=1, column=1, sticky="w", padx=10, pady=5)

        ttk.Label(result_grid, text="Tačnost:", font=("Arial", 10)).grid(
            row=1, column=2, sticky="w", padx=10, pady=5)
        self.floor_conf = ttk.Label(result_grid, text="N/A", font=("Arial", 10))
        self.floor_conf.grid(row=1, column=3, sticky="w", padx=10, pady=5)

        # Prostorija
        ttk.Label(result_grid, text="Prostorija:", font=("Arial", 12, "bold")).grid(
            row=2, column=0, sticky="w", padx=10, pady=5)
        self.room_result = ttk.Label(result_grid, text="N/A",
                                     font=("Arial", 12), foreground="purple")
        self.room_result.grid(row=2, column=1, sticky="w", padx=10, pady=5)

        ttk.Label(result_grid, text="Tačnost:", font=("Arial", 10)).grid(
            row=2, column=2, sticky="w", padx=10, pady=5)
        self.room_conf = ttk.Label(result_grid, text="N/A", font=("Arial", 10))
        self.room_conf.grid(row=2, column=3, sticky="w", padx=10, pady=5)

        # === Log/history tabela ===
        log_frame = ttk.LabelFrame(self.root, text="Historija Lokalizacija", padding=10)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Scrollbar
        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.pack(side="right", fill="y")

        # Treeview za historiju
        columns = ("Vrijeme", "Zgrada", "Sprat", "Prostorija")
        self.history_tree = ttk.Treeview(log_frame, columns=columns, show="headings",
                                         yscrollcommand=scrollbar.set, height=6)
        scrollbar.config(command=self.history_tree.yview)

        for col in columns:
            self.history_tree.heading(col, text=col)
            if col == "Vrijeme":
                self.history_tree.column(col, width=100)
            elif col == "Prostorija":
                self.history_tree.column(col, width=200)
            else:
                self.history_tree.column(col, width=120)

        self.history_tree.pack(fill="both", expand=True)

        # === Status bar ===
        self.status_var = tk.StringVar(value="Učitajte model za početak lokalizacije")
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                              relief="sunken", anchor="w")
        status_bar.pack(fill="x", side="bottom", padx=10, pady=5)

    def load_model(self):
        """Učitavanje istrenirane mreže"""
        filename = filedialog.askopenfilename(
            title="Odaberi istreniranu mrežu",
            filetypes=[("Keras model", "*.keras"), ("All files", "*.*")]
        )

        if not filename:
            return

        try:
            # Učitaj model
            self.model = keras.models.load_model(filename)

            # Učitaj metadata
            metadata_file = filename.replace('.keras', '_metadata.pkl')
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)

            self.scaler = metadata['scaler']
            self.num_buildings = metadata['num_buildings']
            self.num_floors = metadata['num_floors']
            self.num_rooms = metadata['num_rooms']
            self.wifi_cols = metadata['wifi_cols']
            self.encoders = metadata['encoders']

            self.model_loaded = True

            # Ažuriraj GUI
            self.model_label.config(text=f"Model učitan: {filename.split('/')[-1]}",
                                   foreground="green")
            self.start_button.config(state="normal")
            self.status_var.set("Model uspješno učitan. Kliknite 'Lokalizacija' za početak.")

            messagebox.showinfo("Uspjeh", "Model uspješno učitan!")

        except Exception as e:
            messagebox.showerror("Greška",
                               f"Greška pri učitavanju modela:\n{str(e)}")
            self.model_loaded = False

    def scan_wifi_networks(self):
        """Skeniranje WiFi mreža i prikupljanje RSSI podataka"""
        try:
            # Pokušaj sa nmcli komandom
            result = subprocess.run(['nmcli', '-t', '-f', 'SSID,BSSID,SIGNAL',
                                   'dev', 'wifi', 'list'],
                                  capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                networks = {}
                for line in result.stdout.strip().split('\n'):
                    if line:
                        # Split od kraja da izvučemo SIGNAL
                        temp_parts = line.rsplit(':', 1)
                        if len(temp_parts) == 2:
                            signal = temp_parts[1]
                            ssid_bssid = temp_parts[0]

                            # BSSID je poslednji 6 dijelova odvojenih sa ':'
                            all_parts = ssid_bssid.split(':')
                            if len(all_parts) >= 6:
                                bssid = ':'.join(all_parts[-6:])

                                # Konverzija signala u RSSI
                                try:
                                    signal_percent = int(signal)
                                    rssi = (signal_percent // 2) - 100
                                except ValueError:
                                    rssi = -100

                                networks[bssid] = rssi

                return networks

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Ako ne radi, generiši simulirane podatke
        return self.generate_simulated_wifi_data()

    def generate_simulated_wifi_data(self):
        """Generisanje simuliranih WiFi podataka za testiranje"""
        import random

        networks = {}
        num_networks = random.randint(5, 15)

        for i in range(num_networks):
            bssid = ':'.join([f'{random.randint(0, 255):02X}' for _ in range(6)])
            rssi = random.randint(-90, -30)
            networks[bssid] = rssi

        return networks

    def localization_worker(self):
        """Worker thread za periodičnu lokalizaciju"""
        while self.is_localizing:
            try:
                # Skeniranje WiFi mreža
                wifi_data = self.scan_wifi_networks()

                # Kreiranje feature vektora
                # Moramo koristiti iste WAP kolone kao u treniranom modelu
                X_input = []
                for col in self.wifi_cols:
                    # Ekstrahovati BSSID iz imena kolone (npr. WAP_AABBCCDDEEFF -> AA:BB:CC:DD:EE:FF)
                    bssid_clean = col.replace('WAP_', '')
                    # Dodaj ':' svaka 2 karaktera
                    bssid = ':'.join([bssid_clean[i:i+2] for i in range(0, len(bssid_clean), 2)])

                    # Dohvati RSSI ili koristi -100 ako nije detektovan
                    rssi = wifi_data.get(bssid, -100)
                    X_input.append(rssi)

                # Normalizacija
                X_input = np.array(X_input).reshape(1, -1)
                X_normalized = self.scaler.transform(X_input)

                # Predikcija
                predictions = self.model.predict(X_normalized, verbose=0)
                building_pred = predictions[0]
                floor_pred = predictions[1]
                room_pred = predictions[2]

                # Rezultati
                building_id = np.argmax(building_pred[0])
                building_conf = building_pred[0][building_id]
                floor_id = np.argmax(floor_pred[0])
                floor_conf = floor_pred[0][floor_id]
                room_id = np.argmax(room_pred[0])
                room_conf = room_pred[0][room_id]

                # Dekodiranje naziva lokacije
                building_name = self.encoders['building'].classes_[building_id]
                floor_name = self.encoders['floor'].classes_[floor_id]
                room_name = self.encoders['room'].classes_[room_id]

                # Ažuriranje GUI-a (thread-safe)
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.root.after(0, self.update_results,
                              building_id, building_name, building_conf,
                              floor_id, floor_name, floor_conf,
                              room_id, room_name, room_conf,
                              timestamp)

                # Čekanje za sledeće mjerenje
                time.sleep(self.localization_period)

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Greška", f"Greška tokom lokalizacije: {str(e)}"))
                self.root.after(0, self.stop_localization)
                break

    def update_results(self, building_id, building_name, building_conf,
                      floor_id, floor_name, floor_conf,
                      room_id, room_name, room_conf, timestamp):
        """Ažuriranje prikaza rezultata"""
        # Ažuriraj labele
        self.building_result.config(text=f"{building_name}")
        self.building_conf.config(text=f"{building_conf*100:.1f}%")

        self.floor_result.config(text=f"{floor_name}")
        self.floor_conf.config(text=f"{floor_conf*100:.1f}%")

        self.room_result.config(text=f"{room_name}")
        self.room_conf.config(text=f"{room_conf*100:.1f}%")

        # Dodaj u historiju (najnoviji na vrhu)
        self.history_tree.insert("", 0, values=(
            timestamp,
            building_name,
            floor_name,
            room_name
        ))

        # Ograniči broj redova u historiji
        children = self.history_tree.get_children()
        if len(children) > 50:
            self.history_tree.delete(children[-1])

        # Auto-scroll na vrh
        if children:
            self.history_tree.see(children[0])

        # Ažuriraj status
        self.status_var.set(f"Lokalizacija u toku... Poslednje ažuriranje: {timestamp}")

    def start_localization(self):
        """Pokretanje lokalizacije"""
        if not self.model_loaded:
            messagebox.showwarning("Upozorenje", "Model nije učitan!")
            return

        try:
            self.localization_period = int(self.period_var.get())
            if self.localization_period < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Greška", "Period mora biti pozitivan broj")
            return

        # Početak lokalizacije
        self.is_localizing = True
        self.localization_thread = threading.Thread(target=self.localization_worker,
                                                    daemon=True)
        self.localization_thread.start()

        # Ažuriranje GUI-a
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.load_button.config(state="disabled")

        self.status_var.set("Lokalizacija pokrenuta...")

    def stop_localization(self):
        """Zaustavljanje lokalizacije"""
        self.is_localizing = False

        # Ažuriranje GUI-a
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.load_button.config(state="normal")

        self.status_var.set("Lokalizacija zaustavljena")


def main():
    root = tk.Tk()
    app = WiFiLocalizationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
