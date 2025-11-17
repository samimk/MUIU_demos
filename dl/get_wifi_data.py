"""
WiFi Data Collection Tool
=========================
Aplikacija za prikupljanje WiFi RSSI podataka sa anotacijom lokacije
(zgrada, sprat, prostorija) za potrebe treniranja modela za lokalizaciju.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import subprocess
import re
import threading
import time
from datetime import datetime
import csv
import os


class WiFiDataCollector:
    def __init__(self, root):
        self.root = root
        self.root.title("WiFi Data Collector")
        self.root.geometry("900x700")

        # Kreiranje menija
        self.create_menu()

        # Učitavanje konfiguracije
        self.config = self.load_config()

        # Podaci za prikupljanje
        self.collected_data = []
        self.is_collecting = False
        self.collection_thread = None
        self.measurement_period = 5  # sekundi

        # Kreiranje GUI-a
        self.create_gui()

    def load_config(self):
        """Učitavanje konfiguracije iz JSON fajla"""
        config_file = os.path.join(os.path.dirname(__file__), 'wifi_config.json')
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            messagebox.showerror("Greška",
                               f"Konfiguracioni fajl nije pronađen: {config_file}")
            return {"buildings": []}
        except json.JSONDecodeError:
            messagebox.showerror("Greška",
                               "Greška pri parsiranju JSON konfiguracionog fajla")
            return {"buildings": []}

    def create_menu(self):
        """Kreiranje menija"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Help meni
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def show_about(self):
        """Prikaz About dijaloga"""
        messagebox.showinfo("About",
                           "Mašinsko učenje i inteligentno upravljanje\n\n"
                           "Red. prof. dr Samim Konjicija\n\n"
                           "Novembar 2025. godine")

    def create_gui(self):
        """Kreiranje GUI elemenata"""

        # === Sekcija za odabir lokacije ===
        location_frame = ttk.LabelFrame(self.root, text="Lokacija", padding=10)
        location_frame.pack(fill="x", padx=10, pady=5)

        # Zgrada
        ttk.Label(location_frame, text="Zgrada:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.building_var = tk.StringVar()
        self.building_combo = ttk.Combobox(location_frame, textvariable=self.building_var,
                                          state="readonly", width=30)
        self.building_combo['values'] = [b['name'] for b in self.config['buildings']]
        if self.building_combo['values']:
            self.building_combo.current(0)
        self.building_combo.grid(row=0, column=1, padx=5, pady=5)
        self.building_combo.bind('<<ComboboxSelected>>', self.on_building_changed)

        # Sprat
        ttk.Label(location_frame, text="Sprat:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.floor_var = tk.StringVar()
        self.floor_combo = ttk.Combobox(location_frame, textvariable=self.floor_var,
                                       state="readonly", width=30)
        self.floor_combo.grid(row=1, column=1, padx=5, pady=5)
        self.floor_combo.bind('<<ComboboxSelected>>', self.on_floor_changed)

        # Prostorija
        ttk.Label(location_frame, text="Prostorija:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.room_var = tk.StringVar()
        self.room_combo = ttk.Combobox(location_frame, textvariable=self.room_var,
                                      state="readonly", width=30)
        self.room_combo.grid(row=2, column=1, padx=5, pady=5)

        # Inicijalizacija spratova i prostorija
        self.on_building_changed(None)

        # === Sekcija za podešavanja ===
        settings_frame = ttk.LabelFrame(self.root, text="Podešavanja", padding=10)
        settings_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(settings_frame, text="Period mjerenja (sekundi):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.period_var = tk.StringVar(value="5")
        period_spinbox = ttk.Spinbox(settings_frame, from_=1, to=60,
                                    textvariable=self.period_var, width=10)
        period_spinbox.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        # === Kontrole ===
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)

        self.start_button = ttk.Button(control_frame, text="Započni prikupljanje",
                                      command=self.start_collection)
        self.start_button.pack(side="left", padx=5)

        self.stop_button = ttk.Button(control_frame, text="Zaustavi prikupljanje",
                                     command=self.stop_collection, state="disabled")
        self.stop_button.pack(side="left", padx=5)

        self.reset_button = ttk.Button(control_frame, text="Resetiraj prikupljanje",
                                      command=self.reset_collection)
        self.reset_button.pack(side="left", padx=5)

        self.export_button = ttk.Button(control_frame, text="Izvezi podatke",
                                       command=self.export_data)
        self.export_button.pack(side="left", padx=5)

        # === Tabela sa prikupljenim podacima ===
        data_frame = ttk.LabelFrame(self.root, text="Prikupljeni podaci", padding=10)
        data_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Scrollbar
        scrollbar = ttk.Scrollbar(data_frame)
        scrollbar.pack(side="right", fill="y")

        # Treeview za prikaz podataka
        columns = ("Vrijeme", "Zgrada", "Sprat", "Prostorija", "SSID", "BSSID", "RSSI")
        self.data_tree = ttk.Treeview(data_frame, columns=columns, show="headings",
                                     yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.data_tree.yview)

        for col in columns:
            self.data_tree.heading(col, text=col)
            if col == "SSID":
                self.data_tree.column(col, width=150)
            elif col == "BSSID":
                self.data_tree.column(col, width=150)
            elif col == "Vrijeme":
                self.data_tree.column(col, width=80)
            else:
                self.data_tree.column(col, width=100)

        self.data_tree.pack(fill="both", expand=True)

        # === Status bar ===
        self.status_var = tk.StringVar(value="Spremno za prikupljanje podataka")
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                              relief="sunken", anchor="w")
        status_bar.pack(fill="x", side="bottom", padx=10, pady=5)

    def on_building_changed(self, event):
        """Handler za promjenu zgrade"""
        building_name = self.building_var.get()
        building = next((b for b in self.config['buildings'] if b['name'] == building_name), None)

        if building:
            self.floor_combo['values'] = [f['name'] for f in building['floors']]
            if self.floor_combo['values']:
                self.floor_combo.current(0)
                self.on_floor_changed(None)

    def on_floor_changed(self, event):
        """Handler za promjenu sprata"""
        building_name = self.building_var.get()
        floor_name = self.floor_var.get()

        building = next((b for b in self.config['buildings'] if b['name'] == building_name), None)
        if building:
            floor = next((f for f in building['floors'] if f['name'] == floor_name), None)
            if floor:
                self.room_combo['values'] = floor['rooms']
                if self.room_combo['values']:
                    self.room_combo.current(0)

    def scan_wifi_networks(self):
        """Skeniranje WiFi mreža i prikupljanje RSSI podataka"""
        try:
            # Pokušaj sa nmcli komandom
            result = subprocess.run(['nmcli', '-t', '-f', 'SSID,BSSID,SIGNAL',
                                   'dev', 'wifi', 'list'],
                                  capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                networks = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        # nmcli format: SSID:BSSID:SIGNAL
                        # Problem: BSSID sadrži ':' (format XX:XX:XX:XX:XX:XX)
                        # Rješenje: split od kraja da izvučemo SIGNAL, pa onda BSSID

                        # Split od kraja da izvučemo SIGNAL (poslednji dio)
                        temp_parts = line.rsplit(':', 1)
                        if len(temp_parts) == 2:
                            signal = temp_parts[1]
                            ssid_bssid = temp_parts[0]

                            # BSSID je poslednji 6 dijelova odvojenih sa ':' (XX:XX:XX:XX:XX:XX)
                            all_parts = ssid_bssid.split(':')
                            if len(all_parts) >= 6:
                                # Poslednji 6 dijelova su BSSID
                                bssid = ':'.join(all_parts[-6:])
                                # Sve ostalo je SSID
                                ssid = ':'.join(all_parts[:-6]) if len(all_parts) > 6 else "(hidden)"
                                ssid = ssid if ssid else "(hidden)"

                                # Konverzija signala u RSSI (dBm)
                                # nmcli vraća signal u procentima (0-100)
                                # Konverzija: RSSI = signal/2 - 100 (daje raspon -100 do -50 dBm)
                                try:
                                    signal_percent = int(signal)
                                    rssi = (signal_percent // 2) - 100
                                except ValueError:
                                    rssi = -100

                                networks.append({
                                    'ssid': ssid,
                                    'bssid': bssid,
                                    'rssi': rssi
                                })

                return networks

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Alternativa: pokušaj sa iwlist (zahtijeva sudo privilegije)
        try:
            result = subprocess.run(['iwlist', 'scanning'],
                                  capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                networks = []
                current_network = {}

                for line in result.stdout.split('\n'):
                    line = line.strip()

                    if 'Address:' in line and 'Cell' in line:
                        if current_network:
                            networks.append(current_network)
                        current_network = {}
                        match = re.search(r'Address:\s*([0-9A-Fa-f:]+)', line)
                        if match:
                            current_network['bssid'] = match.group(1)

                    elif 'ESSID:' in line:
                        match = re.search(r'ESSID:"([^"]*)"', line)
                        if match:
                            current_network['ssid'] = match.group(1) or "(hidden)"

                    elif 'Signal level=' in line:
                        match = re.search(r'Signal level=(-?\d+)', line)
                        if match:
                            current_network['rssi'] = int(match.group(1))

                if current_network:
                    networks.append(current_network)

                return networks

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Ako ni jedna metoda ne radi, generiši simulirane podatke (za testiranje)
        return self.generate_simulated_wifi_data()

    def generate_simulated_wifi_data(self):
        """Generisanje simuliranih WiFi podataka za testiranje"""
        import random

        networks = []
        num_networks = random.randint(3, 8)

        for i in range(num_networks):
            ssid = f"WiFi_Network_{i+1}"
            bssid = ':'.join([f'{random.randint(0, 255):02X}' for _ in range(6)])
            rssi = random.randint(-90, -30)

            networks.append({
                'ssid': ssid,
                'bssid': bssid,
                'rssi': rssi
            })

        return networks

    def collection_worker(self):
        """Worker thread za periodično prikupljanje podataka"""
        while self.is_collecting:
            try:
                # Validacija lokacije
                if not self.building_var.get() or not self.floor_var.get() or not self.room_var.get():
                    self.root.after(0, lambda: messagebox.showwarning(
                        "Upozorenje", "Molimo odaberite zgradu, sprat i prostoriju"))
                    self.root.after(0, self.stop_collection)
                    break

                # Skeniranje WiFi mreža
                networks = self.scan_wifi_networks()

                # Dodavanje podataka
                timestamp = datetime.now().strftime("%H:%M:%S")
                building = self.building_var.get()
                floor = self.floor_var.get()
                room = self.room_var.get()

                for network in networks:
                    data_entry = {
                        'timestamp': timestamp,
                        'building': building,
                        'floor': floor,
                        'room': room,
                        'ssid': network.get('ssid', ''),
                        'bssid': network.get('bssid', ''),
                        'rssi': network.get('rssi', -100)
                    }

                    self.collected_data.append(data_entry)

                    # Ažuriranje GUI-a (thread-safe)
                    self.root.after(0, self.add_data_to_tree,
                                  timestamp, building, floor, room,
                                  network.get('ssid', ''),
                                  network.get('bssid', ''),
                                  network.get('rssi', -100))

                # Ažuriranje statusa
                self.root.after(0, lambda: self.status_var.set(
                    f"Prikupljanje u toku... Ukupno uzoraka: {len(self.collected_data)} | "
                    f"Poslednje mjerenje: {len(networks)} AP-ova"))

                # Čekanje za sledeće mjerenje
                time.sleep(self.measurement_period)

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Greška", f"Greška tokom prikupljanja podataka: {str(e)}"))
                self.root.after(0, self.stop_collection)
                break

    def add_data_to_tree(self, timestamp, building, floor, room, ssid, bssid, rssi):
        """Dodavanje podataka u treeview (najnoviji na vrhu)"""
        self.data_tree.insert("", 0, values=(timestamp, building, floor, room,
                                            ssid, bssid, rssi))
        # Auto-scroll na vrh (najnoviji podaci)
        children = self.data_tree.get_children()
        if children:
            self.data_tree.see(children[0])

    def start_collection(self):
        """Započinjanje prikupljanja podataka"""
        # Validacija
        if not self.building_var.get() or not self.floor_var.get() or not self.room_var.get():
            messagebox.showwarning("Upozorenje",
                                  "Molimo odaberite zgradu, sprat i prostoriju")
            return

        try:
            self.measurement_period = int(self.period_var.get())
            if self.measurement_period < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Greška", "Period mjerenja mora biti pozitivan broj")
            return

        # Početak prikupljanja
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self.collection_worker, daemon=True)
        self.collection_thread.start()

        # Ažuriranje GUI-a
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.building_combo.config(state="disabled")
        self.floor_combo.config(state="disabled")
        self.room_combo.config(state="disabled")

        self.status_var.set("Prikupljanje pokrenuto...")

    def stop_collection(self):
        """Zaustavljanje prikupljanja podataka"""
        self.is_collecting = False

        # Ažuriranje GUI-a
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.building_combo.config(state="readonly")
        self.floor_combo.config(state="readonly")
        self.room_combo.config(state="readonly")

        self.status_var.set(f"Prikupljanje zaustavljeno. Ukupno uzoraka: {len(self.collected_data)}")

    def reset_collection(self):
        """Resetiranje prikupljenih podataka"""
        if self.is_collecting:
            messagebox.showwarning("Upozorenje",
                                  "Zaustavite prikupljanje prije resetiranja")
            return

        if len(self.collected_data) > 0:
            result = messagebox.askyesno("Potvrda",
                                        "Da li ste sigurni da želite obrisati sve prikupljene podatke?")
            if result:
                self.collected_data.clear()

                # Brisanje iz treeview-a
                for item in self.data_tree.get_children():
                    self.data_tree.delete(item)

                self.status_var.set("Podaci resetirani. Spremno za novo prikupljanje.")
        else:
            messagebox.showinfo("Informacija", "Nema podataka za resetiranje")

    def export_data(self):
        """Izvoz podataka u CSV fajl"""
        if not self.collected_data:
            messagebox.showwarning("Upozorenje", "Nema podataka za izvoz")
            return

        # Dijalog za snimanje fajla
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"wifi_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

        if filename:
            try:
                with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['building', 'floor', 'room', 'ap_bssid', 'ap_ssid', 'rssi', 'timestamp']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    writer.writeheader()
                    for entry in self.collected_data:
                        writer.writerow({
                            'building': entry['building'],
                            'floor': entry['floor'],
                            'room': entry['room'],
                            'ap_bssid': entry['bssid'],
                            'ap_ssid': entry['ssid'],
                            'rssi': entry['rssi'],
                            'timestamp': entry['timestamp']
                        })

                messagebox.showinfo("Uspjeh",
                                  f"Podaci uspješno izvezeni!\n\n"
                                  f"Fajl: {filename}\n"
                                  f"Ukupno uzoraka: {len(self.collected_data)}")

                self.status_var.set(f"Podaci izvezeni u: {filename}")

            except Exception as e:
                messagebox.showerror("Greška",
                                   f"Greška tokom izvoza podataka:\n{str(e)}")


def main():
    root = tk.Tk()
    app = WiFiDataCollector(root)
    root.mainloop()


if __name__ == "__main__":
    main()
