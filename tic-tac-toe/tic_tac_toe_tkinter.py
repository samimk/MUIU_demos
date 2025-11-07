"""
TIC-TAC-TOE ML - Tkinter GUI Vizualizacija
==========================================

Grafička vizualizacija Mitchell-ovog pristupa dizajnu ML sistema.

Korištenje:
    python tic_tac_toe_gui.py

Potrebno: tkinter (uključen u standardnu Python instalaciju)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import random
from typing import List, Tuple, Optional

# ============================================================================
# ML LOGIKA (ista kao prethodno, ali sa GUI callback-ovima)
# ============================================================================

class TicTacToeML:
    """ML sistem za Tic-Tac-Toe sa GUI podrškom"""
    
    def __init__(self):
        self.weights = {
            #'my_pieces': 0.00,
            #'opp_pieces': 0.00,
            #'my_two_in_row': 0.00,
            #'opp_two_in_row': 0.00,
            #'center': 0.00,
            #'corners': 0.00
            'my_pieces': 1.00,
            'opp_pieces': -1.00,
            'my_two_in_row': 5.00,
            'opp_two_in_row': -8.00,
            'center': 2.00,
            'corners': 1.50
        }
        self.stats = {'X': 0, 'O': 0, 'draw': 0}
        self.games_played = 0
        
    def create_empty_board(self):
        return np.full((3, 3), ' ', dtype=str)
    
    def check_winner(self, board):
        for i in range(3):
            if board[i, 0] == board[i, 1] == board[i, 2] != ' ':
                return board[i, 0]
            if board[0, i] == board[1, i] == board[2, i] != ' ':
                return board[0, i]
        if board[0, 0] == board[1, 1] == board[2, 2] != ' ':
            return board[1, 1]
        if board[0, 2] == board[1, 1] == board[2, 0] != ' ':
            return board[1, 1]
        return None
    
    def is_board_full(self, board):
        return not np.any(board == ' ')
    
    def extract_features(self, board, player):
        opponent = 'O' if player == 'X' else 'X'
        features = {
            'my_pieces': 0, 'opp_pieces': 0,
            'my_two_in_row': 0, 'opp_two_in_row': 0,
            'center': 0, 'corners': 0
        }
        
        my_count = np.sum(board == player)
        opp_count = np.sum(board == opponent)
        features['my_pieces'] = my_count
        features['opp_pieces'] = opp_count
        
        if board[1, 1] == player:
            features['center'] = 1
        
        corners = [(0,0), (0,2), (2,0), (2,2)]
        features['corners'] = sum(1 for r, c in corners if board[r, c] == player)
        
        lines = [
            [(0,0), (0,1), (0,2)], [(1,0), (1,1), (1,2)], [(2,0), (2,1), (2,2)],
            [(0,0), (1,0), (2,0)], [(0,1), (1,1), (2,1)], [(0,2), (1,2), (2,2)],
            [(0,0), (1,1), (2,2)], [(0,2), (1,1), (2,0)]
        ]
        
        for line in lines:
            cells = [board[r, c] for r, c in line]
            my_cnt = cells.count(player)
            opp_cnt = cells.count(opponent)
            empty_cnt = cells.count(' ')
            
            if my_cnt == 2 and empty_cnt == 1:
                features['my_two_in_row'] += 1
            if opp_cnt == 2 and empty_cnt == 1:
                features['opp_two_in_row'] += 1
        
        return features
    
    def evaluate_board(self, board, player):
        features = self.extract_features(board, player)
        value = sum(self.weights[key] * features[key] for key in self.weights)
        return value
    
    def get_available_moves(self, board):
        moves = []
        for i in range(3):
            for j in range(3):
                if board[i, j] == ' ':
                    moves.append((i, j))
        return moves
    
    def get_best_move(self, board, player, epsilon=0.0):
        available_moves = self.get_available_moves(board)
        if not available_moves:
            return None
        
        if epsilon > 0 and random.random() < epsilon:
            return random.choice(available_moves)
        
        best_value = -np.inf
        best_move = None
        
        for move in available_moves:
            test_board = board.copy()
            test_board[move[0], move[1]] = player
            value = self.evaluate_board(test_board, player)
            
            if value > best_value:
                best_value = value
                best_move = move
        
        return best_move
    
    def get_move_evaluations(self, board, player):
        """Vraća evaluacije svih poteza za vizualizaciju"""
        available_moves = self.get_available_moves(board)
        evaluations = {}
        
        for move in available_moves:
            test_board = board.copy()
            test_board[move[0], move[1]] = player
            value = self.evaluate_board(test_board, player)
            evaluations[move] = value
        
        return evaluations
    
    def train(self, num_games=100, learning_rate=0.05, epsilon=0.2, 
              progress_callback=None):
        """Trening sa callback-om za progress bar"""
        initial_weights = self.weights.copy()
        
        for game in range(num_games):
            history_X = []
            history_O = []
            board = self.create_empty_board()
            current_player = 'X'
            
            while True:
                features = self.extract_features(board, current_player)
                move = self.get_best_move(board, current_player, epsilon=epsilon)
                
                if move is None:
                    break
                
                if current_player == 'X':
                    history_X.append((board.copy(), features.copy()))
                else:
                    history_O.append((board.copy(), features.copy()))
                
                board[move[0], move[1]] = current_player
                winner = self.check_winner(board)
                
                if winner:
                    self.stats[winner] += 1
                    if winner == 'X':
                        self._update_weights_from_history(history_X, 1.0, learning_rate)
                        self._update_weights_from_history(history_O, -1.0, learning_rate)
                    else:
                        self._update_weights_from_history(history_O, 1.0, learning_rate)
                        self._update_weights_from_history(history_X, -1.0, learning_rate)
                    break
                
                if self.is_board_full(board):
                    self.stats['draw'] += 1
                    self._update_weights_from_history(history_X, 0.1, learning_rate*0.3)
                    self._update_weights_from_history(history_O, 0.1, learning_rate*0.3)
                    break
                
                current_player = 'O' if current_player == 'X' else 'X'
            
            self.games_played += 1
            
            # Update progress
            if progress_callback:
                progress_callback(game + 1, num_games)
        
        return initial_weights
    
    def _update_weights_from_history(self, history, reward, learning_rate):
        for idx, (board_state, features) in enumerate(history):
            time_weight = (idx + 1) / len(history)
            effective_lr = learning_rate * time_weight
            
            for key in self.weights.keys():
                if features[key] > 0:
                    self.weights[key] += effective_lr * reward * features[key]
    
    def reset_weights(self):
        """Reset svih težina na 0.00"""
        for key in self.weights:
            self.weights[key] = 0.00


# ============================================================================
# TKINTER GUI
# ============================================================================

class TicTacToeGUI:
    """Grafička vizualizacija Tic-Tac-Toe ML sistema"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe ML - Mitchell-ov pristup")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        self.ml_system = TicTacToeML()
        self.board = self.ml_system.create_empty_board()
        self.current_player = 'X'
        self.game_over = False
        self.show_evaluations = tk.BooleanVar(value=False)
        
        self.setup_gui()
        self.update_displays()
    
    def setup_gui(self):
        """Kreira GUI elemente"""
        # Glavni kontejner
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # LIJEVA STRANA - Igra
        left_frame = tk.Frame(main_container, bg='white', relief=tk.RAISED, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Naslov
        title_label = tk.Label(left_frame, text="Tic-Tac-Toe ML", 
                               font=('Arial', 24, 'bold'), bg='white', fg='#2c3e50')
        title_label.pack(pady=20)
        
        # Status
        self.status_label = tk.Label(left_frame, text="Na potezu: X (Vi)", 
                                     font=('Arial', 14), bg='white', fg='#27ae60')
        self.status_label.pack(pady=10)
        
        # Tabla
        board_frame = tk.Frame(left_frame, bg='white')
        board_frame.pack(pady=20)
        
        # Fiksna veličina za svako polje
        CELL_SIZE = 120
        
        self.buttons = []
        for i in range(3):
            row_buttons = []
            for j in range(3):
                # Frame kontejner sa fiksnom veličinom
                cell_frame = tk.Frame(board_frame, width=CELL_SIZE, height=CELL_SIZE, 
                                     bg='#ecf0f1', relief=tk.RAISED, bd=3)
                cell_frame.grid(row=i, column=j, padx=5, pady=5)
                cell_frame.grid_propagate(False)  # Ne dozvoli resize
                
                # Button unutar frame-a
                btn = tk.Button(cell_frame, text='', font=('Arial', 36, 'bold'),
                               bg='#ecf0f1', command=lambda r=i, c=j: self.make_move(r, c),
                               relief=tk.FLAT, bd=0, cursor='hand2')
                btn.place(relx=0.5, rely=0.5, anchor='center', relwidth=1, relheight=1)
                row_buttons.append(btn)
            self.buttons.append(row_buttons)
        
        # Evaluacije (opciono)
        eval_check = tk.Checkbutton(left_frame, text="Prikaži evaluacije AI poteza",
                                   variable=self.show_evaluations,
                                   command=self.update_board_display,
                                   font=('Arial', 10), bg='white')
        eval_check.pack(pady=10)
        
        # Kontrole igre
        game_controls = tk.Frame(left_frame, bg='white')
        game_controls.pack(pady=20)
        
        tk.Button(game_controls, text="Nova igra", command=self.new_game,
                 font=('Arial', 12, 'bold'), bg='#3498db', fg='white',
                 padx=20, pady=10, relief=tk.RAISED).pack(side=tk.LEFT, padx=5)
        
        # DESNA STRANA - ML komponente
        right_frame = tk.Frame(main_container, bg='white', relief=tk.RAISED, bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Notebook za tabove
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # TAB 1: Težine
        weights_tab = tk.Frame(notebook, bg='white')
        notebook.add(weights_tab, text='3. Reprezentacija')
        self.setup_weights_tab(weights_tab)
        
        # TAB 2: Trening
        training_tab = tk.Frame(notebook, bg='white')
        notebook.add(training_tab, text='4. Algoritam učenja')
        self.setup_training_tab(training_tab)
        
        # TAB 3: Statistika
        stats_tab = tk.Frame(notebook, bg='white')
        notebook.add(stats_tab, text='Statistika')
        self.setup_stats_tab(stats_tab)
        
        # TAB 4: Objašnjenje
        explain_tab = tk.Frame(notebook, bg='white')
        notebook.add(explain_tab, text='Mitchell-ov pristup')
        self.setup_explain_tab(explain_tab)
    
    def setup_weights_tab(self, parent):
        """Tab za prikaz težina"""
        tk.Label(parent, text="Parametri modela (težine)", 
                font=('Arial', 14, 'bold'), bg='white').pack(pady=10)
        
        formula = tk.Label(parent, 
                          text="V(board) = w₁·x₁ + w₂·x₂ + w₃·x₃ + w₄·x₄ + w₅·x₅ + w₆·x₆",
                          font=('Courier', 10), bg='#ecf0f1', fg='#2c3e50')
        formula.pack(pady=10, padx=10, fill=tk.X)
        
        self.weight_labels = {}
        weights_frame = tk.Frame(parent, bg='white')
        weights_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        weight_names = [
            ('my_pieces', 'w₁: Broj mojih figura'),
            ('opp_pieces', 'w₂: Broj protivničkih figura'),
            ('my_two_in_row', 'w₃: Moje "2 u redu"'),
            ('opp_two_in_row', 'w₄: Protivničke "2 u redu"'),
            ('center', 'w₅: Kontrola centra'),
            ('corners', 'w₆: Kontrola uglova')
        ]
        
        for key, name in weight_names:
            frame = tk.Frame(weights_frame, bg='white')
            frame.pack(pady=5, padx=20, fill=tk.X)
            
            tk.Label(frame, text=name, font=('Arial', 11), 
                    bg='white', anchor='w').pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            label = tk.Label(frame, text="0.500", font=('Courier', 11, 'bold'),
                           bg='#ecf0f1', fg='#27ae60', width=8)
            label.pack(side=tk.RIGHT)
            self.weight_labels[key] = label
        
        # Reset button
        tk.Button(parent, text="Reset težina (sve → 0)",
                 command=self.reset_weights,
                 font=('Arial', 10), bg='#e74c3c', fg='white',
                 padx=10, pady=5).pack(pady=20)
    
    def setup_training_tab(self, parent):
        """Tab za trening"""
        tk.Label(parent, text="Self-play trening", 
                font=('Arial', 14, 'bold'), bg='white').pack(pady=10)
        
        tk.Label(parent, text="Algoritam uči kroz igre protiv samog sebe",
                font=('Arial', 10), bg='white', fg='#7f8c8d').pack(pady=5)
        
        # Parametri
        params_frame = tk.LabelFrame(parent, text="Parametri treninga", 
                                     font=('Arial', 11, 'bold'), bg='white')
        params_frame.pack(pady=20, padx=20, fill=tk.X)
        
        # Broj igara
        tk.Label(params_frame, text="Broj igara:", bg='white').grid(row=0, column=0, 
                                                                     sticky='w', padx=10, pady=5)
        self.games_var = tk.StringVar(value="100")
        tk.Entry(params_frame, textvariable=self.games_var, width=10).grid(row=0, column=1, 
                                                                            padx=10, pady=5)
        
        # Learning rate
        tk.Label(params_frame, text="Learning rate:", bg='white').grid(row=1, column=0, 
                                                                        sticky='w', padx=10, pady=5)
        self.lr_var = tk.StringVar(value="0.05")
        tk.Entry(params_frame, textvariable=self.lr_var, width=10).grid(row=1, column=1, 
                                                                         padx=10, pady=5)
        
        # Epsilon
        tk.Label(params_frame, text="Epsilon (exploration):", bg='white').grid(row=2, column=0, 
                                                                                sticky='w', padx=10, pady=5)
        self.epsilon_var = tk.StringVar(value="0.2")
        tk.Entry(params_frame, textvariable=self.epsilon_var, width=10).grid(row=2, column=1, 
                                                                              padx=10, pady=5)
        
        # Progress bar
        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(parent, variable=self.progress_var, 
                                           maximum=100, length=300)
        self.progress_bar.pack(pady=20)
        
        self.progress_label = tk.Label(parent, text="", font=('Arial', 10), bg='white')
        self.progress_label.pack(pady=5)
        
        # Train button
        self.train_button = tk.Button(parent, text="Treniraj AI",
                                      command=self.train_ai,
                                      font=('Arial', 12, 'bold'), bg='#27ae60', fg='white',
                                      padx=30, pady=15, relief=tk.RAISED)
        self.train_button.pack(pady=20)
    
    def setup_stats_tab(self, parent):
        """Tab za statistiku"""
        tk.Label(parent, text="Statistika igara", 
                font=('Arial', 14, 'bold'), bg='white').pack(pady=10)
        
        self.stats_frame = tk.Frame(parent, bg='white')
        self.stats_frame.pack(pady=20, fill=tk.BOTH, expand=True)
        
        # Kreiraj labele za statistiku
        stats_items = [
            ('X pobjede:', 'x_wins', '#3498db'),
            ('O pobjede:', 'o_wins', '#e74c3c'),
            ('Nerješeno:', 'draws', '#95a5a6'),
            ('Ukupno treninga:', 'total_games', '#2c3e50')
        ]
        
        self.stats_labels = {}
        for text, key, color in stats_items:
            frame = tk.Frame(self.stats_frame, bg='white')
            frame.pack(pady=10, padx=20, fill=tk.X)
            
            tk.Label(frame, text=text, font=('Arial', 12), 
                    bg='white', anchor='w').pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            label = tk.Label(frame, text="0", font=('Arial', 16, 'bold'),
                           bg='white', fg=color, width=6)
            label.pack(side=tk.RIGHT)
            self.stats_labels[key] = label
    
    def setup_explain_tab(self, parent):
        """Tab za objašnjenje Mitchell-ovog pristupa"""
        # Scrollable text
        canvas = tk.Canvas(parent, bg='white')
        scrollbar = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        explanation_text = """
Mitchell-ov pristup dizajnu ML sistema
=======================================

Dizajn ML sistema zahtijeva 4 ključne odluke:

[1] TIP ISKUSTVA (E)
--------------------
• Kako će sistem učiti?
• Tic-Tac-Toe: Self-play (igra protiv sebe)
• Dame: Self-play ili protiv eksperta
• Pitanje: Odakle dolaze podaci za učenje?

[2] CILJNA FUNKCIJA (T)
------------------------
• Šta sistem treba naučiti?
• V(board) -> procjena koliko je pozicija dobra
• Veća vrijednost = bolja pozicija za igrača
• Pitanje: Šta je cilj učenja?

[3] REPREZENTACIJA
-------------------
• Kako predstaviti naučeno znanje?
• Ne možemo tabelu (previše pozicija!)
• Koristimo funkcijsku aproksimaciju:

  V(board) = w1*x1 + w2*x2 + ... + w6*x6

  gdje su:
  x1 = broj mojih figura
  x2 = broj protivničkih figura  
  x3 = moje "2 u redu"
  x4 = protivničke "2 u redu"
  x5 = kontrola centra
  x6 = kontrola uglova

• w1, w2, ... su PARAMETRI koje učimo!
• Pitanje: Kako reprezentovati znanje?

[4] ALGORITAM UČENJA
--------------------
• Kako ažurirati parametre w1, w2, ...?
• Outcome-based learning:
  - Igraj self-play partije
  - Pobjednički potezi -> pojačaj težine
  - Gubitnički potezi -> smanji težine
• Kasniji potezi važniji (bliže ishodu)
• Pitanje: Kako poboljšati performanse?

========================================

ANALOGIJA SA KONTROLNIM SISTEMIMA:
----------------------------------
Iskustvo -> Senzorski podaci
Ciljna funkcija -> Kontrolna politika
Reprezentacija -> Model sistema
Algoritam -> Optimizacioni metod
        """
        
        text_widget = tk.Text(scrollable_frame, wrap=tk.WORD, font=('Courier', 10),
                             bg='white', padx=20, pady=20, bd=0)
        text_widget.insert('1.0', explanation_text)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def make_move(self, row, col):
        """Igrač pravi potez"""
        if self.game_over or self.board[row, col] != ' ' or self.current_player != 'X':
            return
        
        # Igrač potez
        self.board[row, col] = 'X'
        self.update_board_display()
        
        winner = self.ml_system.check_winner(self.board)
        if winner:
            self.end_game(winner)
            return
        
        if self.ml_system.is_board_full(self.board):
            self.end_game('draw')
            return
        
        self.current_player = 'O'
        self.status_label.config(text="Na potezu: O (AI razmišlja...)")
        self.root.update()
        
        # AI potez
        self.root.after(500, self.ai_move)
    
    def ai_move(self):
        """AI pravi potez"""
        move = self.ml_system.get_best_move(self.board, 'O')
        if move:
            self.board[move[0], move[1]] = 'O'
            self.update_board_display()
            
            winner = self.ml_system.check_winner(self.board)
            if winner:
                self.end_game(winner)
                return
            
            if self.ml_system.is_board_full(self.board):
                self.end_game('draw')
                return
        
        self.current_player = 'X'
        self.status_label.config(text="Na potezu: X (Vi)", fg='#27ae60')
    
    def end_game(self, result):
        """Završava igru"""
        self.game_over = True
        
        if result == 'draw':
            self.status_label.config(text="= Nerješeno!", fg='#95a5a6')
            messagebox.showinfo("Kraj igre", "Igra je nerješena!")
        else:
            winner_text = "Vi" if result == 'X' else "AI"
            color = '#27ae60' if result == 'X' else '#e74c3c'
            self.status_label.config(text=f"* Pobjednik: {winner_text}!", fg=color)
            messagebox.showinfo("Kraj igre", f"Pobjednik: {winner_text}!")
        
        # Update statistike
        if result != 'draw':
            self.ml_system.stats[result] += 1
        else:
            self.ml_system.stats['draw'] += 1
        
        self.update_stats_display()
    
    def new_game(self):
        """Nova igra"""
        self.board = self.ml_system.create_empty_board()
        self.current_player = 'X'
        self.game_over = False
        self.status_label.config(text="Na potezu: X (Vi)", fg='#27ae60')
        self.update_board_display()
    
    def update_board_display(self):
        """Ažurira prikaz table"""
        evaluations = {}
        if self.show_evaluations.get() and not self.game_over:
            evaluations = self.ml_system.get_move_evaluations(self.board, 'O')
        
        for i in range(3):
            for j in range(3):
                cell = self.board[i, j]
                btn = self.buttons[i][j]
                
                if cell == 'X':
                    btn.config(text='X', fg='#3498db', bg='#d6eaf8', 
                              font=('Arial', 48, 'bold'), state=tk.DISABLED)
                elif cell == 'O':
                    btn.config(text='O', fg='#e74c3c', bg='#fadbd8',
                              font=('Arial', 48, 'bold'), state=tk.DISABLED)
                else:
                    # Prazno polje
                    if (i, j) in evaluations:
                        # Prikaži evaluaciju
                        eval_val = evaluations[(i, j)]
                        btn.config(text=f'{eval_val:.1f}', fg='#7f8c8d', bg='#ecf0f1',
                                  font=('Arial', 16, 'bold'), state=tk.NORMAL)
                    else:
                        # Potpuno prazno
                        btn.config(text='', bg='#ecf0f1', state=tk.NORMAL,
                                  font=('Arial', 48, 'bold'), fg='black')
    
    def update_displays(self):
        """Ažurira sve prikaze"""
        self.update_weight_display()
        self.update_stats_display()
    
    def update_weight_display(self):
        """Ažurira prikaz težina"""
        for key, label in self.weight_labels.items():
            value = self.ml_system.weights[key]
            color = '#27ae60' if value > 0 else '#e74c3c' if value < 0 else '#95a5a6'
            label.config(text=f"{value:+.3f}", fg=color)
    
    def update_stats_display(self):
        """Ažurira prikaz statistike"""
        self.stats_labels['x_wins'].config(text=str(self.ml_system.stats['X']))
        self.stats_labels['o_wins'].config(text=str(self.ml_system.stats['O']))
        self.stats_labels['draws'].config(text=str(self.ml_system.stats['draw']))
        self.stats_labels['total_games'].config(text=str(self.ml_system.games_played))
    
    def train_ai(self):
        """Pokreće trening AI-a"""
        try:
            num_games = int(self.games_var.get())
            learning_rate = float(self.lr_var.get())
            epsilon = float(self.epsilon_var.get())
        except ValueError:
            messagebox.showerror("Greška", "Neispravni parametri!")
            return
        
        self.train_button.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.progress_label.config(text="Trening u toku...")
        
        def progress_callback(current, total):
            progress = int((current / total) * 100)
            self.progress_var.set(progress)
            self.progress_label.config(text=f"Trening: {current}/{total} igara")
            self.root.update_idletasks()
        
        # Sačuvaj početne težine
        initial_weights = self.ml_system.weights.copy()
        
        # Trening
        self.ml_system.train(num_games, learning_rate, epsilon, progress_callback)
        
        # Prikaži promjene
        changes = []
        for key in self.ml_system.weights:
            delta = self.ml_system.weights[key] - initial_weights[key]
            changes.append(f"{key}: {initial_weights[key]:.3f} → {self.ml_system.weights[key]:.3f} (Δ {delta:+.3f})")
        
        self.progress_label.config(text="[OK] Trening završen!")
        self.update_displays()
        self.train_button.config(state=tk.NORMAL)
        
        messagebox.showinfo("Trening završen", 
                           f"Završeno {num_games} igara!\n\n" +
                           "Promjene težina:\n" + "\n".join(changes))
    
    def reset_weights(self):
        """Reset težina na početne vrijednosti"""
        if messagebox.askyesno("Potvrda", "Da li želite resetovati sve težine na 0?"):
            self.ml_system.reset_weights()
            self.update_weight_display()
            messagebox.showinfo("Reset", "Težine resetovane na 0")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Pokreće GUI aplikaciju"""
    root = tk.Tk()
    app = TicTacToeGUI(root)
    
    # Center window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    main()
