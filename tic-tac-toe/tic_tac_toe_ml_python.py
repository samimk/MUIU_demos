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
            #'my_pieces': 1.0,           # w1: broj mojih figura
            #'opp_pieces': -1.0,         # w2: broj protivničkih figura
            #'my_two_in_row': 5.0,       # w3: moje 2 u redu
            #'opp_two_in_row': -8.0,     # w4: protivničke 2 u redu
            #'center': 2.0,              # w5: kontrola centra
            #'corners': 1.5              # w6: kontrola uglova
        }
        
        self.stats = {'X': 0, 'O': 0, 'draw': 0}
        self.games_played = 0
        
    def create_empty_board(self) -> np.ndarray:
        """Kreira praznu tablu 3x3"""
        return np.full((3, 3), ' ', dtype=str)
    
    def print_board(self, board: np.ndarray):
        """Ispisuje tablu u konzoli"""
        print("\n  0   1   2")
        for i, row in enumerate(board):
            print(f"{i} {row[0]} | {row[1]} | {row[2]}")
            if i < 2:
                print("  ---------")
        print()
    
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
                     verbose: bool = False, epsilon: float = 0.0) -> Tuple[int, int]:
        """
        Bira najbolji potez koristeći evaluacionu funkciju.
        
        Za svaki mogući potez:
        - Simulira potez
        - Evaluira novu poziciju
        - Bira potez sa najvećom vrijednošću
        
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
        evaluations = []
        
        for move in available_moves:
            # Simuliraj potez
            test_board = board.copy()
            test_board[move[0], move[1]] = player
            
            # Evaluiraj poziciju
            value = self.evaluate_board(test_board, player)
            evaluations.append((move, value))
            
            if value > best_value:
                best_value = value
                best_move = move
        
        if verbose:
            print(f"\nEvaluacija poteza za {player}:")
            for move, val in sorted(evaluations, key=lambda x: x[1], reverse=True):
                print(f"  Pozicija {move}: V = {val:.2f}")
            print(f"Izabran potez: {best_move} (V = {best_value:.2f})")
        
        return best_move
    
    def play_game(self, player1: str = 'human', player2: str = 'ai',
                 verbose: bool = True) -> str:
        """
        1. TIP ISKUSTVA - Igra jednu partiju
        
        Args:
            player1: 'human' ili 'ai'
            player2: 'human' ili 'ai'
            verbose: Da li ispisivati detalje
        
        Returns:
            str: 'X', 'O', ili 'draw'
        """
        board = self.create_empty_board()
        current_player = 'X'
        
        while True:
            if verbose:
                self.print_board(board)
                print(f"Na potezu: {current_player}")
            
            # Odabir poteza
            if (current_player == 'X' and player1 == 'human') or \
               (current_player == 'O' and player2 == 'human'):
                # Ljudski igrač
                while True:
                    try:
                        row = int(input("Red (0-2): "))
                        col = int(input("Kolona (0-2): "))
                        if board[row, col] == ' ':
                            break
                        print("To polje je zauzeto!")
                    except (ValueError, IndexError):
                        print("Nevalidan unos! Pokušaj ponovo.")
                move = (row, col)
            else:
                # AI igrač
                move = self.get_best_move(board, current_player, verbose=verbose)
                if verbose:
                    print(f"AI bira: {move}")
            
            # Izvršavanje poteza
            board[move[0], move[1]] = current_player
            
            # Provjera pobjednika
            winner = self.check_winner(board)
            if winner:
                if verbose:
                    self.print_board(board)
                    print(f"🎉 Pobjednik: {winner}!")
                self.stats[winner] += 1
                return winner
            
            # Provjera nerješenog
            if self.is_board_full(board):
                if verbose:
                    self.print_board(board)
                    print("⚖️  Nerješeno!")
                self.stats['draw'] += 1
                return 'draw'
            
            # Sljedeći igrač
            current_player = 'O' if current_player == 'X' else 'X'
    
    def train(self, num_games: int = 100, learning_rate: float = 0.01, epsilon: float = 0.2):
        """
        4. ALGORITAM UČENJA - Self-play sa outcome-based learning
        
        AI igra protiv samog sebe i ažurira težine na osnovu ishoda.
        Koristi jednostavan outcome-based pristup:
        - Pobjednički potezi → povećaj težine features koji su bili prisutni
        - Gubitnički potezi → smanji težine features koji su bili prisutni
        
        Args:
            num_games: Broj igara za trening
            learning_rate: Stopa učenja (obično 0.001 - 0.1)
            epsilon: Vjerovatnoća random poteza tokom treninga (exploration)
        """
        print(f"\n🧠 Trening: {num_games} igara self-play...")
        print(f"   Stopa učenja (learning rate): {learning_rate}")
        print(f"   Exploration rate (epsilon): {epsilon}")
        
        initial_weights = self.weights.copy()
        
        for game in range(num_games):
            # Čuvaj historiju pozicija i poteza za oba igrača
            history_X = []  # (board_state, features)
            history_O = []
            
            board = self.create_empty_board()
            current_player = 'X'
            
            # Igraj jednu partiju i čuvaj historiju
            while True:
                # Snimi features prije poteza
                features = self.extract_features(board, current_player)
                
                # Odaberi potez (sa eksploracijom tokom treninga)
                move = self.get_best_move(board, current_player, verbose=False, epsilon=epsilon)
                if move is None:
                    break
                
                # Snimi u historiju
                if current_player == 'X':
                    history_X.append((board.copy(), features.copy()))
                else:
                    history_O.append((board.copy(), features.copy()))
                
                # Izvršavanje poteza
                board[move[0], move[1]] = current_player
                
                # Provjera pobjednika
                winner = self.check_winner(board)
                if winner:
                    self.stats[winner] += 1
                    
                    # AŽURIRANJE TEŽINA na osnovu ishoda
                    if winner == 'X':
                        # X je pobijedio - pojačaj features koji su X koristio
                        self._update_weights_from_history(history_X, reward=1.0, learning_rate=learning_rate)
                        # X je pobijedio - smanji features koji su O koristio
                        self._update_weights_from_history(history_O, reward=-1.0, learning_rate=learning_rate)
                    else:  # winner == 'O'
                        # O je pobijedio
                        self._update_weights_from_history(history_O, reward=1.0, learning_rate=learning_rate)
                        self._update_weights_from_history(history_X, reward=-1.0, learning_rate=learning_rate)
                    break
                
                # Provjera nerješenog
                if self.is_board_full(board):
                    self.stats['draw'] += 1
                    # Nerješeno - malo pojačaj features iz kasnih poteza
                    self._update_weights_from_history(history_X, reward=0.1, learning_rate=learning_rate*0.3)
                    self._update_weights_from_history(history_O, reward=0.1, learning_rate=learning_rate*0.3)
                    break
                
                # Sljedeći igrač
                current_player = 'O' if current_player == 'X' else 'X'
            
            self.games_played += 1
        
        print(f"✅ Trening završen!")
        print(f"\nPromjene u težinama:")
        for key in self.weights:
            change = self.weights[key] - initial_weights[key]
            arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
            print(f"  {key:20s}: {initial_weights[key]:7.3f} → {self.weights[key]:7.3f} "
                  f"({arrow} Δ = {change:+.3f})")
    
    def _update_weights_from_history(self, history: List, reward: float, learning_rate: float):
        """
        Pomoćna funkcija za ažuriranje težina na osnovu historije poteza.
        
        Ideja: Features koji su bili prisutni u pobjedničkim/gubitničkim pozicijama
        treba da budu pojačani/smanjeni.
        
        Args:
            history: Lista (board, features) tuple-ova
            reward: +1 za pobjedu, -1 za poraz, ~0 za nerješeno
            learning_rate: Stopa učenja
        """
        # Ažuriraj samo na osnovu kasnijih poteza (bliže ishodu)
        # Daj veći značaj kasnijim potezima
        for idx, (board_state, features) in enumerate(history):
            # Težinski faktor - kasniji potezi važniji
            time_weight = (idx + 1) / len(history)
            effective_lr = learning_rate * time_weight
            
            # Ažuriraj svaku težinu na osnovu prisustva feature-a
            for key in self.weights.keys():
                if features[key] > 0:  # Ako je feature bio prisutan
                    # Pomjeri težinu u smjeru nagrade
                    self.weights[key] += effective_lr * reward * features[key]
    
    def print_weights(self):
        """Ispisuje trenutne težine"""
        print("\n📊 Trenutne težine (parametri modela):")
        print("V(board) = w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6")
        print()
        for i, (key, value) in enumerate(self.weights.items(), 1):
            print(f"  w{i} ({key:20s}): {value:7.3f}")
    
    def print_stats(self):
        """Ispisuje statistiku igara"""
        total = sum(self.stats.values())
        print(f"\n📈 Statistika ({total} igara):")
        print(f"  X pobjede:  {self.stats['X']:3d} ({self.stats['X']/total*100:.1f}%)")
        print(f"  O pobjede:  {self.stats['O']:3d} ({self.stats['O']/total*100:.1f}%)")
        print(f"  Nerješeno:  {self.stats['draw']:3d} ({self.stats['draw']/total*100:.1f}%)")
        print(f"  Ukupno treninga: {self.games_played} igara")


def demonstrate_mitchell_approach():
    """
    Demonstracija Mitchell-ovog pristupa dizajnu ML sistema
    """
    print("="*60)
    print("TIC-TAC-TOE: Mitchell-ov pristup ML sistemu")
    print("="*60)
    
    ml_system = TicTacToeML()
    
    # Prikaz komponenti
    print("\n📋 4 KLJUČNE KOMPONENTE ML SISTEMA:\n")
    
    print("1️⃣  TIP ISKUSTVA (E):")
    print("   - Self-play: AI igra protiv samog sebe")
    print("   - Uči iz ishoda igara (pobjeda/poraz/nerješeno)")
    
    print("\n2️⃣  CILJNA FUNKCIJA (T):")
    print("   - V(board) → procjena vrijednosti pozicije")
    print("   - Cilj: naučiti koje pozicije vode pobjedi")
    
    print("\n3️⃣  REPREZENTACIJA:")
    print("   - Linearna funkcija features:")
    ml_system.print_weights()
    
    print("\n4️⃣  ALGORITAM UČENJA:")
    print("   - Self-play + reinforcement")
    print("   - Ažuriranje težina na osnovu ishoda")
    
    return ml_system


def interactive_demo():
    """Interaktivni demo za nastavu"""
    ml_system = demonstrate_mitchell_approach()
    
    while True:
        print("\n" + "="*60)
        print("OPCIJE:")
        print("  1. Igraj protiv AI")
        print("  2. Treniraj AI (100 igara)")
        print("  3. Prikaži težine")
        print("  4. Prikaži statistiku")
        print("  5. Reset sistema")
        print("  6. Izlaz")
        print("="*60)
        
        choice = input("\nIzbor: ").strip()
        
        if choice == '1':
            print("\n🎮 Ti si X, AI je O")
            ml_system.play_game(player1='human', player2='ai', verbose=True)
            ml_system.print_stats()
            
        elif choice == '2':
            # Možeš podesiti parametre
            lr = 0.05  # learning rate - eksperimentiraj sa 0.01 - 0.1
            eps = 0.2  # exploration rate - eksperimentiraj sa 0.1 - 0.3
            ml_system.train(num_games=100, learning_rate=lr, epsilon=eps)
            ml_system.print_weights()
            ml_system.print_stats()
            
        elif choice == '3':
            ml_system.print_weights()
            
        elif choice == '4':
            ml_system.print_stats()
            
        elif choice == '5':
            ml_system = demonstrate_mitchell_approach()
            print("\n✅ Sistem resetovan!")
            
        elif choice == '6':
            print("\n👋 Hvala na korištenju!")
            break
        
        else:
            print("❌ Nevalidan izbor!")


# ============================================================================
# PRIMJERI KORIŠTENJA
# ============================================================================

if __name__ == "__main__":
    # Za demonstraciju u nastavi
    interactive_demo()
    
    # Ili za Jupyter notebook:
    """
    # Kreiraj sistem
    ml_system = demonstrate_mitchell_approach()
    
    # Igraj
    ml_system.play_game(player1='human', player2='ai')
    
    # Treniraj
    ml_system.train(100)
    
    # Pregledaj težine
    ml_system.print_weights()
    
    # Statistika
    ml_system.print_stats()
    """
