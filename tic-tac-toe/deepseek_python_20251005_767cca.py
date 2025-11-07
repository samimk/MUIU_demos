"""
TIC-TAC-TOE: Mašinsko učenje - Mitchell-ov pristup
===================================================

Ilustracija 4 komponente dizajna ML sistema:
1. Tip iskustva (E)
2. Ciljna funkcija (T) 
3. Reprezentacija
4. Algoritam učenja

Autor: Za kurs "Machine Learning and Intelligent Control"
"""

import numpy as np
import random
from typing import List, Tuple, Optional
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

class TicTacToeML:
    """
    ML sistem za učenje Tic-Tac-Toe igre kroz self-play.
    
    Komponente (Mitchell):
    - Iskustvo: Self-play igre
    - Ciljna funkcija: V(board) - evaluacija pozicije
    - Reprezentacija: Linearna kombinacija features
    - Algoritam: Učenje težina kroz reinforcement
    """
    
    def __init__(self):
        """Inicijalizacija težina (parametara modela)"""
        # 3. REPREZENTACIJA - Parametri modela
        self.weights = {
            'my_pieces': -0.001,           # w1: broj mojih figura
            'opp_pieces': -0.001,         # w2: broj protivničkih figura
            'my_two_in_row': -0.001,       # w3: moje 2 u redu
            'opp_two_in_row': -0.001,     # w4: protivničke 2 u redu
            'center': -0.001,              # w5: kontrola centra
            'corners': -0.001              # w6: kontrola uglova
        }
        
        self.stats = {'X': 0, 'O': 0, 'draw': 0}
        self.games_played = 0
        
    def create_empty_board(self) -> np.ndarray:
        """Kreira praznu tablu 3x3"""
        return np.full((3, 3), ' ', dtype=str)
    
    def check_winner(self, board: np.ndarray) -> Optional[str]:
        """Provjerava pobjednika"""
        # Redovi, kolone, dijagonale
        for i in range(3):
            # Redovi
            if board[i, 0] == board[i, 1] == board[i, 2] != ' ':
                return board[i, 0]
            # Kolone
            if board[0, i] == board[1, i] == board[2, i] != ' ':
                return board[0, i]
        
        # Dijagonale
        if board[0, 0] == board[1, 1] == board[2, 2] != ' ':
            return board[1, 1]
        if board[0, 2] == board[1, 1] == board[2, 0] != ' ':
            return board[1, 1]
        
        return None
    
    def is_board_full(self, board: np.ndarray) -> bool:
        """Provjerava da li je tabla puna"""
        return not np.any(board == ' ')
    
    def extract_features(self, board: np.ndarray, player: str) -> dict:
        """
        2. CILJNA FUNKCIJA - Ekstraktuje features za evaluaciju
        
        Features (x_i):
        - x1: Broj mojih figura
        - x2: Broj protivničkih figura
        - x3: Broj mojih "2 u redu"
        - x4: Broj protivničkih "2 u redu"
        - x5: Da li kontrolišem centar
        - x6: Koliko uglova kontrolišem
        """
        opponent = 'O' if player == 'X' else 'X'
        
        features = {
            'my_pieces': 0,
            'opp_pieces': 0,
            'my_two_in_row': 0,
            'opp_two_in_row': 0,
            'center': 0,
            'corners': 0
        }
        
        # Brojanje figura
        my_count = np.sum(board == player)
        opp_count = np.sum(board == opponent)
        features['my_pieces'] = my_count
        features['opp_pieces'] = opp_count
        
        # Centar (pozicija [1,1])
        if board[1, 1] == player:
            features['center'] = 1
        
        # Uglovi
        corners = [(0,0), (0,2), (2,0), (2,2)]
        features['corners'] = sum(1 for r, c in corners if board[r, c] == player)
        
        # Dvije u redu (sa praznim trećim mjestom)
        lines = [
            # Redovi
            [(0,0), (0,1), (0,2)],
            [(1,0), (1,1), (1,2)],
            [(2,0), (2,1), (2,2)],
            # Kolone
            [(0,0), (1,0), (2,0)],
            [(0,1), (1,1), (2,1)],
            [(0,2), (1,2), (2,2)],
            # Dijagonale
            [(0,0), (1,1), (2,2)],
            [(0,2), (1,1), (2,0)]
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
    
    def evaluate_board(self, board: np.ndarray, player: str) -> float:
        """
        2. CILJNA FUNKCIJA - V(board)
        
        Procjenjuje vrijednost pozicije za datog igrača.
        V(board) = w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6
        
        Returns:
            float: Procjena vrijednosti (-inf do +inf)
                  Veća vrijednost = bolja pozicija za igrača
        """
        features = self.extract_features(board, player)
        
        # Linearna kombinacija features
        value = (
            self.weights['my_pieces'] * features['my_pieces'] +
            self.weights['opp_pieces'] * features['opp_pieces'] +
            self.weights['my_two_in_row'] * features['my_two_in_row'] +
            self.weights['opp_two_in_row'] * features['opp_two_in_row'] +
            self.weights['center'] * features['center'] +
            self.weights['corners'] * features['corners']
        )
        
        return value
    
    def get_available_moves(self, board: np.ndarray) -> List[Tuple[int, int]]:
        """Vraća listu dostupnih poteza"""
        moves = []
        for i in range(3):
            for j in range(3):
                if board[i, j] == ' ':
                    moves.append((i, j))
        return moves
    
    def get_best_move(self, board: np.ndarray, player: str, 
                     epsilon: float = 0.0) -> Tuple[int, int]:
        """
        Bira najbolji potez koristeći evaluacionu funkciju.
        
        Args:
            epsilon: Vjerovatnoća random poteza (exploration), 0.0 = greedy
        """
        available_moves = self.get_available_moves(board)
        
        if not available_moves:
            return None
        
        # Epsilon-greedy strategija za trening
        if epsilon > 0 and random.random() < epsilon:
            # Exploration - random potez
            return random.choice(available_moves)
        
        # Exploitation - najbolji potez
        best_value = -np.inf
        best_move = None
        
        for move in available_moves:
            # Simuliraj potez
            test_board = board.copy()
            test_board[move[0], move[1]] = player
            
            # Evaluiraj poziciju
            value = self.evaluate_board(test_board, player)
            
            if value > best_value:
                best_value = value
                best_move = move
        
        return best_move
    
    def train(self, num_games: int = 100, learning_rate: float = 0.01, epsilon: float = 0.2):
        """
        4. ALGORITAM UČENJA - Self-play sa outcome-based learning
        """
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
                        self._update_weights_from_history(history_X, reward=1.0, learning_rate=learning_rate)
                        self._update_weights_from_history(history_O, reward=-1.0, learning_rate=learning_rate)
                    else:
                        self._update_weights_from_history(history_O, reward=1.0, learning_rate=learning_rate)
                        self._update_weights_from_history(history_X, reward=-1.0, learning_rate=learning_rate)
                    break
                
                if self.is_board_full(board):
                    self.stats['draw'] += 1
                    self._update_weights_from_history(history_X, reward=0.1, learning_rate=learning_rate*0.3)
                    self._update_weights_from_history(history_O, reward=0.1, learning_rate=learning_rate*0.3)
                    break
                
                current_player = 'O' if current_player == 'X' else 'X'
            
            self.games_played += 1
        
        return initial_weights
    
    def _update_weights_from_history(self, history: List, reward: float, learning_rate: float):
        """Pomoćna funkcija za ažuriranje težina na osnovu historije poteza."""
        for idx, (board_state, features) in enumerate(history):
            time_weight = (idx + 1) / len(history) if history else 1.0
            effective_lr = learning_rate * time_weight
            
            for key in self.weights.keys():
                if features[key] > 0:
                    self.weights[key] += effective_lr * reward * features[key]
    
    def print_weights(self):
        """Ispisuje trenutne težine"""
        result = "Trenutne težine (parametri modela):\n"
        result += "V(board) = w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6\n\n"
        for i, (key, value) in enumerate(self.weights.items(), 1):
            result += f"  w{i} ({key:20s}): {value:7.3f}\n"
        return result
    
    def print_stats(self):
        """Ispisuje statistiku igara"""
        total = sum(self.stats.values())
        result = f"Statistika ({total} igara):\n"
        result += f"  X pobjede:  {self.stats['X']:3d} ({self.stats['X']/total*100:.1f}%)\n"
        result += f"  O pobjede:  {self.stats['O']:3d} ({self.stats['O']/total*100:.1f}%)\n"
        result += f"  Nerješeno:  {self.stats['draw']:3d} ({self.stats['draw']/total*100:.1f}%)\n"
        result += f"  Ukupno treninga: {self.games_played} igara"
        return result


class TicTacToeGUI:
    """GUI aplikacija za Tic-Tac-Toe ML sistem"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe ML - Mitchell-ov pristup")
        self.root.geometry("800x700")
        
        self.ml_system = TicTacToeML()
        self.current_player = 'X'
        self.board = self.ml_system.create_empty_board()
        self.game_active = False
        
        self.setup_gui()
        self.update_display()
    
    def setup_gui(self):
        """Postavlja GUI elemente"""
        # Glavni frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Konfiguracija grid-a
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Naslov
        title_label = ttk.Label(main_frame, 
                               text="TIC-TAC-TOE: Mitchell-ov pristup ML sistemu",
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Frame za tablu
        board_frame = ttk.LabelFrame(main_frame, text="Tabla", padding="10")
        board_frame.grid(row=1, column=0, padx=(0, 10), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Kreiranje dugmića za tablu
        self.buttons = []
        for i in range(3):
            row_buttons = []
            for j in range(3):
                btn = tk.Button(board_frame, text=' ', font=('Arial', 20, 'bold'),
                               width=4, height=2,
                               command=lambda row=i, col=j: self.human_move(row, col))
                btn.grid(row=i, column=j, padx=2, pady=2)
                row_buttons.append(btn)
            self.buttons.append(row_buttons)
        
        # Frame za kontrolu
        control_frame = ttk.LabelFrame(main_frame, text="Kontrola", padding="10")
        control_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status display
        self.status_var = tk.StringVar(value="Klikni 'Nova igra' da počneš!")
        status_label = ttk.Label(control_frame, textvariable=self.status_var,
                                font=('Arial', 12), wraplength=300)
        status_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Dugmad za kontrolu
        ttk.Button(control_frame, text="Nova igra", 
                  command=self.new_game).grid(row=1, column=0, pady=5, sticky=tk.EW)
        ttk.Button(control_frame, text="AI potez", 
                  command=self.ai_move).grid(row=1, column=1, pady=5, sticky=tk.EW)
        
        # Frame za trening
        train_frame = ttk.LabelFrame(control_frame, text="Trening AI", padding="5")
        train_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=tk.EW)
        
        ttk.Label(train_frame, text="Broj igara:").grid(row=0, column=0, sticky=tk.W)
        self.games_var = tk.StringVar(value="100")
        games_entry = ttk.Entry(train_frame, textvariable=self.games_var, width=10)
        games_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(train_frame, text="Stopa učenja:").grid(row=1, column=0, sticky=tk.W)
        self.lr_var = tk.StringVar(value="0.05")
        lr_entry = ttk.Entry(train_frame, textvariable=self.lr_var, width=10)
        lr_entry.grid(row=1, column=1, padx=5)
        
        ttk.Button(train_frame, text="Treniraj AI", 
                  command=self.train_ai).grid(row=2, column=0, columnspan=2, pady=5, sticky=tk.EW)
        
        # Frame za informacije
        info_frame = ttk.LabelFrame(control_frame, text="Informacije", padding="5")
        info_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky=tk.EW)
        
        ttk.Button(info_frame, text="Prikaži težine", 
                  command=self.show_weights).grid(row=0, column=0, pady=2, sticky=tk.EW)
        ttk.Button(info_frame, text="Prikaži statistiku", 
                  command=self.show_stats).grid(row=1, column=0, pady=2, sticky=tk.EW)
        ttk.Button(info_frame, text="Reset sistem", 
                  command=self.reset_system).grid(row=2, column=0, pady=2, sticky=tk.EW)
        
        # Text area za output
        output_frame = ttk.LabelFrame(main_frame, text="ML Informacije", padding="10")
        output_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        main_frame.rowconfigure(2, weight=1)
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, width=80, height=15)
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Postavi početni output
        self.update_output("Dobrodošli u Tic-Tac-Toe ML sistem!\n\n" +
                          "Mitchell-ove 4 komponente ML sistema:\n" +
                          "1. Tip iskustva (E): Self-play igre\n" +
                          "2. Ciljna funkcija (T): V(board) - evaluacija pozicije\n" +
                          "3. Reprezentacija: Linearna kombinacija features\n" +
                          "4. Algoritam: Učenje težina kroz reinforcement\n\n" +
                          "Klikni 'Nova igra' da počneš!")
    
    def update_output(self, text):
        """Ažurira output text area"""
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(1.0, text)
    
    def append_output(self, text):
        """Dodaje tekst u output text area"""
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.see(tk.END)
    
    def update_display(self):
        """Ažurira prikaz table"""
        for i in range(3):
            for j in range(3):
                text = self.board[i, j]
                color = 'black'
                if text == 'X':
                    color = 'blue'
                elif text == 'O':
                    color = 'red'
                self.buttons[i][j].config(text=text, fg=color, state='normal')
    
    def new_game(self):
        """Pokreće novu igru"""
        self.board = self.ml_system.create_empty_board()
        self.current_player = 'X'
        self.game_active = True
        self.status_var.set("Ti si X! Na potezu si.")
        self.update_display()
        self.append_output("--- Nova igra započeta ---")
    
    def human_move(self, row, col):
        """Obrada ljudskog poteza"""
        if not self.game_active or self.board[row, col] != ' ':
            return
        
        # Postavi ljudski potez
        self.board[row, col] = self.current_player
        self.update_display()
        self.append_output(f"Ti ({self.current_player}): pozicija ({row}, {col})")
        
        # Provjeri kraj igre
        if self.check_game_over():
            return
        
        # Prebaci na AI
        self.current_player = 'O'
        self.status_var.set("AI (O) razmišlja...")
        self.root.after(500, self.ai_move)  # Mali delay za bolji UX
    
    def ai_move(self):
        """AI potez"""
        if not self.game_active or self.current_player != 'O':
            return
        
        move = self.ml_system.get_best_move(self.board, 'O')
        if move:
            row, col = move
            self.board[row, col] = 'O'
            self.update_display()
            self.append_output(f"AI (O): pozicija ({row}, {col})")
            
            # Evaluiraj poziciju
            value = self.ml_system.evaluate_board(self.board, 'O')
            features = self.ml_system.extract_features(self.board, 'O')
            self.append_output(f"  Evaluacija: V(board) = {value:.2f}")
            
            if self.check_game_over():
                return
        
        self.current_player = 'X'
        self.status_var.set("Tvoj potez (X)!")
    
    def check_game_over(self):
        """Provjerava kraj igre i ažurira stanje"""
        winner = self.ml_system.check_winner(self.board)
        
        if winner:
            self.game_active = False
            self.status_var.set(f"Pobjednik: {winner}!")
            self.ml_system.stats[winner] += 1
            self.append_output(f"POBJEDNIK: {winner}!")
            messagebox.showinfo("Kraj igre", f"Pobjednik: {winner}!")
            return True
        
        if self.ml_system.is_board_full(self.board):
            self.game_active = False
            self.status_var.set("Nerješeno!")
            self.ml_system.stats['draw'] += 1
            self.append_output("NERJESENO!")
            messagebox.showinfo("Kraj igre", "Nerješeno!")
            return True
        
        return False
    
    def train_ai(self):
        """Pokreće trening AI"""
        try:
            num_games = int(self.games_var.get())
            learning_rate = float(self.lr_var.get())
            
            self.append_output(f"\nPOCETAK TRENINGA: {num_games} igara...")
            self.append_output(f"   Stopa učenja: {learning_rate}")
            
            initial_weights = self.ml_system.train(num_games, learning_rate)
            
            self.append_output("TRENING ZAVRSEN!")
            self.append_output("\nPromjene u težinama:")
            for key in self.ml_system.weights:
                change = self.ml_system.weights[key] - initial_weights[key]
                arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
                self.append_output(f"  {key:20s}: {initial_weights[key]:7.3f} → {self.ml_system.weights[key]:7.3f} "
                                  f"({arrow} Δ = {change:+.3f})")
            
        except ValueError as e:
            messagebox.showerror("Greška", "Nevalidan unos za trening parametre!")
    
    def show_weights(self):
        """Prikazuje težine u output area"""
        weights_text = self.ml_system.print_weights()
        self.update_output(weights_text)
    
    def show_stats(self):
        """Prikazuje statistiku u output area"""
        stats_text = self.ml_system.print_stats()
        self.update_output(stats_text)
    
    def reset_system(self):
        """Resetuje ML sistem"""
        self.ml_system = TicTacToeML()
        self.update_output("Sistem resetovan!\n\n" +
                          "Mitchell-ove 4 komponente ML sistema:\n" +
                          "1. Tip iskustva (E): Self-play igre\n" +
                          "2. Ciljna funkcija (T): V(board) - evaluacija pozicije\n" +
                          "3. Reprezentacija: Linearna kombinacija features\n" +
                          "4. Algoritam: Učenje težina kroz reinforcement")
        self.status_var.set("Sistem resetovan. Klikni 'Nova igra'!")


def main():
    """Glavna funkcija za pokretanje GUI aplikacije"""
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
