import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import random
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class QLearningGrid:
    def __init__(self, size: int = 5):
        self.size = size
        self.Q = np.zeros((size, size, 4))  # 4 actions: up, down, left, right
        self.goal_state = (size - 1, size - 1)
        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }

    def reset(self, size: int):
        """Reset Q-table with new size"""
        self.size = size
        self.Q = np.zeros((size, size, 4))
        self.goal_state = (size - 1, size - 1)

    def get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Get next state given current state and action"""
        delta = self.actions[action]
        next_state = (state[0] + delta[0], state[1] + delta[1])

        # Check boundaries
        if (0 <= next_state[0] < self.size and
            0 <= next_state[1] < self.size):
            return next_state
        return state  # Stay in same position if out of bounds

    def get_reward(self, state: Tuple[int, int]) -> float:
        """Get reward for reaching a state"""
        if state == self.goal_state:
            return 100.0
        return -1.0

    def is_terminal(self, state: Tuple[int, int]) -> bool:
        """Check if state is terminal"""
        return state == self.goal_state

    def choose_action(self, state: Tuple[int, int], epsilon: float = 0.1) -> int:
        """Epsilon-greedy action selection"""
        if random.random() < epsilon:
            return random.randint(0, 3)
        else:
            return np.argmax(self.Q[state[0], state[1], :])

    def update_q(self, state: Tuple[int, int], action: int,
                 next_state: Tuple[int, int], reward: float,
                 alpha: float, gamma: float):
        """Update Q-value using Q-learning update rule"""
        current_q = self.Q[state[0], state[1], action]
        max_next_q = np.max(self.Q[next_state[0], next_state[1], :])
        new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
        self.Q[state[0], state[1], action] = new_q

    def get_best_action(self, state: Tuple[int, int]) -> int:
        """Get best action for a state (greedy)"""
        return np.argmax(self.Q[state[0], state[1], :])


class QLearningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Q-Learning Visualizer")

        # Default parameters
        self.grid_size = 5
        self.cell_size = 80
        self.agent_pos = None
        self.q_learning = QLearningGrid(self.grid_size)

        # Training parameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.animation_speed = 50  # ms

        # UI state
        self.is_training = False
        self.is_running_policy = False
        self.is_policy_step_mode = False  # For step-by-step policy execution
        self.show_q_values = tk.BooleanVar(value=False)
        self.show_arrows = tk.BooleanVar(value=True)
        self.random_start = tk.BooleanVar(value=True)
        self.show_progress_graph = tk.BooleanVar(value=False)
        self.manual_start_state = None

        # Statistics
        self.episode_count = 0
        self.total_episodes = 100
        self.episode_steps_history = []  # Track steps per episode

        # Graph components
        self.graph_window = None
        self.figure = None
        self.ax = None
        self.canvas_graph = None

        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface"""
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Instructions", command=self.show_instructions)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)

        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Grid size
        ttk.Label(control_frame, text="Grid Size:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.size_var = tk.IntVar(value=self.grid_size)
        size_spin = tk.Spinbox(control_frame, from_=3, to=10, textvariable=self.size_var, width=10)
        size_spin.grid(row=0, column=1, sticky=tk.W, pady=2)
        ttk.Button(control_frame, text="Resize", command=self.resize_grid).grid(row=0, column=2, padx=5, pady=2)

        # Learning rate
        ttk.Label(control_frame, text="Learning Rate (α):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.alpha_var = tk.DoubleVar(value=self.alpha)
        alpha_entry = ttk.Entry(control_frame, textvariable=self.alpha_var, width=10)
        alpha_entry.grid(row=1, column=1, sticky=tk.W, pady=2)

        # Discount factor
        ttk.Label(control_frame, text="Discount Factor (γ):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.gamma_var = tk.DoubleVar(value=self.gamma)
        gamma_entry = ttk.Entry(control_frame, textvariable=self.gamma_var, width=10)
        gamma_entry.grid(row=2, column=1, sticky=tk.W, pady=2)

        # Animation speed
        ttk.Label(control_frame, text="Animation Speed (ms):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.speed_var = tk.IntVar(value=self.animation_speed)
        speed_spin = tk.Spinbox(control_frame, from_=10, to=1000, textvariable=self.speed_var, width=10)
        speed_spin.grid(row=3, column=1, sticky=tk.W, pady=2)

        # Episodes
        ttk.Label(control_frame, text="Episodes:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.episodes_var = tk.IntVar(value=self.total_episodes)
        episodes_entry = ttk.Entry(control_frame, textvariable=self.episodes_var, width=10)
        episodes_entry.grid(row=4, column=1, sticky=tk.W, pady=2)

        # Random start checkbox
        ttk.Checkbutton(control_frame, text="Random Start Position",
                       variable=self.random_start).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=2)

        # Show Q-values checkbox
        ttk.Checkbutton(control_frame, text="Show Q-Values",
                       variable=self.show_q_values,
                       command=self.redraw_grid).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=2)

        # Show arrows checkbox
        ttk.Checkbutton(control_frame, text="Show Policy Arrows",
                       variable=self.show_arrows,
                       command=self.redraw_grid).grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=2)

        # Show progress graph checkbox
        ttk.Checkbutton(control_frame, text="Show Progress Graph",
                       variable=self.show_progress_graph,
                       command=self.toggle_progress_graph).grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=2)

        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=9, column=0, columnspan=3, pady=10)

        self.train_btn = ttk.Button(button_frame, text="Start Training", command=self.start_training)
        self.train_btn.grid(row=0, column=0, padx=2)

        self.stop_btn = ttk.Button(button_frame, text="Stop", command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=2)

        ttk.Button(button_frame, text="Reset Q-Table", command=self.reset_q_table).grid(row=0, column=2, padx=2)

        ttk.Button(button_frame, text="Run Policy", command=self.run_policy).grid(row=1, column=0, padx=2, pady=5)

        self.step_policy_btn = ttk.Button(button_frame, text="Step Policy", command=self.step_policy_once)
        self.step_policy_btn.grid(row=1, column=1, padx=2, pady=5)

        self.reset_policy_btn = ttk.Button(button_frame, text="Reset Policy", command=self.reset_policy_run, state=tk.DISABLED)
        self.reset_policy_btn.grid(row=1, column=2, padx=2, pady=5)

        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready", foreground="blue")
        self.status_label.grid(row=10, column=0, columnspan=3, pady=5)

        # Canvas for grid
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        self.canvas = tk.Canvas(canvas_frame, bg="white")
        self.canvas.pack()

        self.draw_grid()
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def draw_grid(self):
        """Draw the grid and all elements"""
        canvas_size = self.grid_size * self.cell_size
        self.canvas.config(width=canvas_size, height=canvas_size)
        self.canvas.delete("all")

        # Draw cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                # Goal state in green
                if (i, j) == self.q_learning.goal_state:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="lightgreen", outline="black", width=2)
                    self.canvas.create_text(x1 + self.cell_size // 2, y1 + 20,
                                          text="GOAL", font=("Arial", 12, "bold"), fill="darkgreen")
                else:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="black")

                # Show Q-values if enabled
                if self.show_q_values.get() and (i, j) != self.q_learning.goal_state:
                    q_vals = self.q_learning.Q[i, j, :]
                    max_q = np.max(q_vals)
                    font_size = max(8, min(10, self.cell_size // 10))

                    # Up
                    self.canvas.create_text(x1 + self.cell_size // 2, y1 + 12,
                                          text=f"{q_vals[0]:.1f}",
                                          font=("Arial", font_size),
                                          fill="red" if q_vals[0] == max_q else "gray")
                    # Down
                    self.canvas.create_text(x1 + self.cell_size // 2, y2 - 12,
                                          text=f"{q_vals[1]:.1f}",
                                          font=("Arial", font_size),
                                          fill="red" if q_vals[1] == max_q else "gray")
                    # Left
                    self.canvas.create_text(x1 + 12, y1 + self.cell_size // 2,
                                          text=f"{q_vals[2]:.1f}",
                                          font=("Arial", font_size),
                                          fill="red" if q_vals[2] == max_q else "gray")
                    # Right
                    self.canvas.create_text(x2 - 12, y1 + self.cell_size // 2,
                                          text=f"{q_vals[3]:.1f}",
                                          font=("Arial", font_size),
                                          fill="red" if q_vals[3] == max_q else "gray")

                # Show policy arrow if enabled
                if self.show_arrows.get() and (i, j) != self.q_learning.goal_state:
                    best_action = self.q_learning.get_best_action((i, j))
                    cx = x1 + self.cell_size // 2
                    cy = y1 + self.cell_size // 2
                    arrow_len = self.cell_size // 3

                    if best_action == 0:  # up
                        self.canvas.create_line(cx, cy, cx, cy - arrow_len,
                                              arrow=tk.LAST, fill="blue", width=2)
                    elif best_action == 1:  # down
                        self.canvas.create_line(cx, cy, cx, cy + arrow_len,
                                              arrow=tk.LAST, fill="blue", width=2)
                    elif best_action == 2:  # left
                        self.canvas.create_line(cx, cy, cx - arrow_len, cy,
                                              arrow=tk.LAST, fill="blue", width=2)
                    elif best_action == 3:  # right
                        self.canvas.create_line(cx, cy, cx + arrow_len, cy,
                                              arrow=tk.LAST, fill="blue", width=2)

        # Draw agent if exists
        if self.agent_pos is not None:
            self.draw_agent(self.agent_pos)

    def draw_agent(self, pos: Tuple[int, int]):
        """Draw agent at given position"""
        i, j = pos
        x = j * self.cell_size + self.cell_size // 2
        y = i * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 4

        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                               fill="red", outline="darkred", width=2, tags="agent")

    def redraw_grid(self):
        """Redraw the entire grid"""
        self.draw_grid()

    def resize_grid(self):
        """Resize the grid"""
        new_size = self.size_var.get()
        if 3 <= new_size <= 10:
            self.grid_size = new_size
            self.q_learning.reset(new_size)
            self.agent_pos = None
            self.manual_start_state = None
            self.draw_grid()
            self.status_label.config(text=f"Grid resized to {new_size}x{new_size}")

    def reset_q_table(self):
        """Reset Q-table to zeros"""
        self.q_learning.Q = np.zeros((self.grid_size, self.grid_size, 4))
        self.episode_count = 0
        self.episode_steps_history = []
        self.agent_pos = None
        self.draw_grid()
        if self.show_progress_graph.get():
            self.update_progress_graph()
        self.status_label.config(text="Q-table reset")

    def on_canvas_click(self, event):
        """Handle canvas click to set manual start position"""
        if not self.is_training and not self.is_running_policy:
            j = event.x // self.cell_size
            i = event.y // self.cell_size

            if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
                self.manual_start_state = (i, j)
                self.agent_pos = (i, j)
                self.draw_grid()
                self.status_label.config(text=f"Start position set to ({i}, {j})")

    def start_training(self):
        """Start Q-learning training"""
        # Reset step mode if active
        if self.is_policy_step_mode:
            self.reset_policy_run()

        self.is_training = True
        self.episode_count = 0
        self.total_episodes = self.episodes_var.get()
        self.alpha = self.alpha_var.get()
        self.gamma = self.gamma_var.get()
        self.animation_speed = self.speed_var.get()

        self.train_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.step_policy_btn.config(state=tk.DISABLED)

        self.run_episode()

    def stop_training(self):
        """Stop training"""
        self.is_training = False
        self.train_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.step_policy_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Training stopped")

    def run_episode(self):
        """Run one training episode"""
        if not self.is_training or self.episode_count >= self.total_episodes:
            self.is_training = False
            self.train_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.step_policy_btn.config(state=tk.NORMAL)
            self.status_label.config(text=f"Training complete! {self.episode_count} episodes")
            self.draw_grid()
            return

        # Initialize starting position
        if self.random_start.get():
            # Random start position (not goal)
            while True:
                start_state = (random.randint(0, self.grid_size - 1),
                             random.randint(0, self.grid_size - 1))
                if start_state != self.q_learning.goal_state:
                    break
        else:
            # Use manual start if set, otherwise (0,0)
            start_state = self.manual_start_state if self.manual_start_state else (0, 0)

        self.current_state = start_state
        self.agent_pos = start_state
        self.episode_count += 1
        self.steps_in_episode = 0

        self.status_label.config(text=f"Training Episode {self.episode_count}/{self.total_episodes}")
        self.step_episode()

    def step_episode(self):
        """Execute one step in the current episode"""
        if not self.is_training:
            return

        # Check if reached goal or too many steps
        if self.q_learning.is_terminal(self.current_state) or self.steps_in_episode > 100:
            # Episode finished, record steps and start next one
            self.episode_steps_history.append(self.steps_in_episode)
            if self.show_progress_graph.get():
                self.update_progress_graph()
            self.root.after(self.animation_speed, self.run_episode)
            return

        # Choose action
        action = self.q_learning.choose_action(self.current_state, self.epsilon)

        # Take action
        next_state = self.q_learning.get_next_state(self.current_state, action)
        reward = self.q_learning.get_reward(next_state)

        # Update Q-value
        self.q_learning.update_q(self.current_state, action, next_state,
                                reward, self.alpha, self.gamma)

        # Move agent
        self.current_state = next_state
        self.agent_pos = next_state
        self.steps_in_episode += 1

        # Redraw
        self.draw_grid()

        # Schedule next step
        self.root.after(self.animation_speed, self.step_episode)

    def run_policy(self):
        """Run the learned policy from a starting position"""
        if self.is_training or self.is_running_policy or self.is_policy_step_mode:
            return

        # Determine start position
        if not self.random_start.get() and self.manual_start_state:
            start_state = self.manual_start_state
        elif not self.random_start.get():
            start_state = (0, 0)
        else:
            while True:
                start_state = (random.randint(0, self.grid_size - 1),
                             random.randint(0, self.grid_size - 1))
                if start_state != self.q_learning.goal_state:
                    break

        self.current_state = start_state
        self.agent_pos = start_state
        self.is_running_policy = True
        self.policy_steps = 0

        self.train_btn.config(state=tk.DISABLED)
        self.step_policy_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Running learned policy...")

        self.step_policy()

    def step_policy(self):
        """Execute one step of the learned policy"""
        if not self.is_running_policy:
            return

        # Check if reached goal or too many steps
        if self.q_learning.is_terminal(self.current_state) or self.policy_steps > 100:
            self.is_running_policy = False
            self.train_btn.config(state=tk.NORMAL)
            self.step_policy_btn.config(state=tk.NORMAL)
            if self.q_learning.is_terminal(self.current_state):
                self.status_label.config(text=f"Goal reached in {self.policy_steps} steps!")
            else:
                self.status_label.config(text="Maximum steps exceeded")
            return

        # Choose best action (greedy)
        action = self.q_learning.get_best_action(self.current_state)

        # Take action
        next_state = self.q_learning.get_next_state(self.current_state, action)

        # Move agent
        self.current_state = next_state
        self.agent_pos = next_state
        self.policy_steps += 1

        # Redraw
        self.draw_grid()

        # Schedule next step
        self.root.after(self.animation_speed, self.step_policy)

    def toggle_progress_graph(self):
        """Toggle the progress graph window"""
        if self.show_progress_graph.get():
            self.create_progress_graph()
        else:
            self.close_progress_graph()

    def create_progress_graph(self):
        """Create a new window with progress graph"""
        if self.graph_window is None or not tk.Toplevel.winfo_exists(self.graph_window):
            self.graph_window = tk.Toplevel(self.root)
            self.graph_window.title("Training Progress")
            self.graph_window.geometry("600x400")

            # Create matplotlib figure
            self.figure = Figure(figsize=(6, 4), dpi=100)
            self.ax = self.figure.add_subplot(111)
            self.ax.set_xlabel('Episode')
            self.ax.set_ylabel('Steps to Goal')
            self.ax.set_title('Learning Progress')
            self.ax.grid(True, alpha=0.3)

            # Create canvas
            self.canvas_graph = FigureCanvasTkAgg(self.figure, master=self.graph_window)
            self.canvas_graph.draw()
            self.canvas_graph.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Handle window close
            self.graph_window.protocol("WM_DELETE_WINDOW", self.on_graph_window_close)

            self.update_progress_graph()

    def update_progress_graph(self):
        """Update the progress graph with current data"""
        if self.ax is not None and len(self.episode_steps_history) > 0:
            self.ax.clear()
            self.ax.set_xlabel('Episode')
            self.ax.set_ylabel('Steps to Goal')
            self.ax.set_title('Learning Progress')
            self.ax.grid(True, alpha=0.3)

            episodes = range(1, len(self.episode_steps_history) + 1)
            self.ax.plot(episodes, self.episode_steps_history, 'b-', linewidth=1, alpha=0.5)

            # Add moving average if enough data
            if len(self.episode_steps_history) >= 10:
                window = 10
                moving_avg = np.convolve(self.episode_steps_history,
                                        np.ones(window)/window, mode='valid')
                avg_episodes = range(window, len(self.episode_steps_history) + 1)
                self.ax.plot(avg_episodes, moving_avg, 'r-', linewidth=2,
                           label=f'{window}-episode moving average')
                self.ax.legend()

            self.canvas_graph.draw()

    def close_progress_graph(self):
        """Close the progress graph window"""
        if self.graph_window is not None:
            self.graph_window.destroy()
            self.graph_window = None
            self.figure = None
            self.ax = None
            self.canvas_graph = None

    def on_graph_window_close(self):
        """Handle graph window close event"""
        self.show_progress_graph.set(False)
        self.close_progress_graph()

    def step_policy_once(self):
        """Execute one step of the policy (for step-by-step execution)"""
        # If not in step mode, initialize it
        if not self.is_policy_step_mode:
            if self.is_training or self.is_running_policy:
                return

            # Determine start position
            if not self.random_start.get() and self.manual_start_state:
                start_state = self.manual_start_state
            elif not self.random_start.get():
                start_state = (0, 0)
            else:
                while True:
                    start_state = (random.randint(0, self.grid_size - 1),
                                 random.randint(0, self.grid_size - 1))
                    if start_state != self.q_learning.goal_state:
                        break

            self.current_state = start_state
            self.agent_pos = start_state
            self.policy_steps = 0
            self.is_policy_step_mode = True

            # Update UI
            self.train_btn.config(state=tk.DISABLED)
            self.reset_policy_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Step-by-step policy mode: Click 'Step Policy' to continue")
            self.draw_grid()
            return

        # Check if reached goal or too many steps
        if self.q_learning.is_terminal(self.current_state) or self.policy_steps > 100:
            self.is_policy_step_mode = False
            self.train_btn.config(state=tk.NORMAL)
            self.reset_policy_btn.config(state=tk.DISABLED)
            if self.q_learning.is_terminal(self.current_state):
                self.status_label.config(text=f"Goal reached in {self.policy_steps} steps!")
            else:
                self.status_label.config(text="Maximum steps exceeded")
            return

        # Choose best action (greedy)
        action = self.q_learning.get_best_action(self.current_state)

        # Take action
        next_state = self.q_learning.get_next_state(self.current_state, action)

        # Move agent
        self.current_state = next_state
        self.agent_pos = next_state
        self.policy_steps += 1

        # Redraw
        self.draw_grid()
        self.status_label.config(text=f"Step {self.policy_steps}: at position {self.current_state}")

    def reset_policy_run(self):
        """Reset the step-by-step policy execution"""
        self.is_policy_step_mode = False
        self.is_running_policy = False
        self.agent_pos = None
        self.train_btn.config(state=tk.NORMAL)
        self.reset_policy_btn.config(state=tk.DISABLED)
        self.draw_grid()
        self.status_label.config(text="Policy run reset")

    def show_instructions(self):
        """Show instructions dialog"""
        instructions_window = tk.Toplevel(self.root)
        instructions_window.title("Uputstvo za korištenje")
        instructions_window.geometry("700x600")
        instructions_window.resizable(True, True)

        # Create scrollable text widget
        frame = ttk.Frame(instructions_window, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Text widget
        text_widget = tk.Text(frame, wrap=tk.WORD, yscrollcommand=scrollbar.set,
                             font=("Arial", 10), padx=10, pady=10)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)

        instructions_text = """UPUTSTVO ZA KORIŠTENJE Q-LEARNING VISUALIZER-a

═══════════════════════════════════════════════════════════════════

1. OSNOVNI KONCEPTI

Q-Learning je algoritam mašinskog učenja koji omogućava agentu da nauči
optimalnu politiku za navigaciju kroz grid. Agent pokušava doći od početne
pozicije do CILJA (zeleno polje u donjem desnom uglu).

═══════════════════════════════════════════════════════════════════

2. PARAMETRI

• Grid Size: Veličina mreže (3-10). Veća mreža = teži problem.

• Learning Rate (α): Brzina učenja (0-1). Veće vrijednosti = brže učenje,
  ali možda manje stabilno.

• Discount Factor (γ): Koliko agent cijeni buduće nagrade (0-1).
  Veće vrijednosti = agent više planira unaprijed.

• Animation Speed: Brzina animacije u milisekundama. Manje = brže.

• Episodes: Broj epizoda treninga. Jedna epizoda = od starta do cilja.

═══════════════════════════════════════════════════════════════════

3. OPCIJE

☑ Random Start Position: Agent startuje sa random pozicije.
  Isključite za fiksni start (kliknite na grid da postavite poziciju).

☑ Show Q-Values: Prikazuje Q-vrijednosti za sve akcije u svakom polju.
  Crvene vrijednosti = najbolja akcija za to polje.

☑ Show Policy Arrows: Prikazuje plave strelice koje pokazuju naučenu
  politiku (najbolju akciju) za svako polje.

☑ Show Progress Graph: Otvara prozor sa grafikom koji pokazuje napredak
  treninga - broj koraka potrebnih da se dostigne cilj po epizodi.

═══════════════════════════════════════════════════════════════════

4. DUGMAD I FUNKCIJE

• START TRAINING: Pokreće trening za zadati broj epizoda.
  Agent će se kretati i učiti optimalnu politiku.

• STOP: Zaustavlja trenutni trening.

• RESET Q-TABLE: Briše sve naučeno i resetuje Q-tabelu na nulu.

• RUN POLICY: Automatski izvršava naučenu politiku od starta do cilja.
  Koristi se nakon treninga da se vidi šta je agent naučio.

• STEP POLICY: Izvršava naučenu politiku KORAK PO KORAK.
  - Prvi klik: postavlja agenta na start poziciju
  - Svaki sljedeći klik: pomjera agenta za jedan korak
  - Odlično za detaljnu analizu naučene politike

• RESET POLICY: Resetuje step-by-step izvršavanje policy-a.

• RESIZE: Mijenja veličinu grid-a na novu vrijednost.

═══════════════════════════════════════════════════════════════════

5. TIPIČAN TOK RADA

1. Podesite parametre (npr. 100 epizoda, α=0.1, γ=0.9)
2. Uključite "Show Progress Graph" da pratite napredak
3. Kliknite "Start Training"
4. Pratite kako agent uči (strelice će pokazivati bolju politiku)
5. Nakon treninga, koristite "Run Policy" ili "Step Policy" da vidite
   naučenu politiku u akciji

═══════════════════════════════════════════════════════════════════

6. GRAFIK NAPREDOVANJA

Plava linija pokazuje broj koraka po epizodi.
Crvena linija (moving average) pokazuje trend - trebala bi opadati
kako agent uči efikasniju politiku.

═══════════════════════════════════════════════════════════════════

7. POSTAVLJANJE POČETNE POZICIJE

1. Isključite "Random Start Position"
2. Kliknite na bilo koje polje u grid-u
3. Agent će startovati sa te pozicije u sljedećoj epizodi/policy run

═══════════════════════════════════════════════════════════════════

8. HELP MENI

• Instructions: Ovo uputstvo
• About: Informacije o programu i autoru

═══════════════════════════════════════════════════════════════════

NAPOMENA: Za najbolje rezultate, pokrenite barem 50-100 epizoda treninga
prije nego što testirate naučenu politiku.
"""

        text_widget.insert("1.0", instructions_text)
        text_widget.config(state=tk.DISABLED)  # Make read-only

        # Close button
        close_btn = ttk.Button(instructions_window, text="Zatvori",
                              command=instructions_window.destroy)
        close_btn.pack(pady=10)

    def show_about(self):
        """Show About dialog"""
        about_text = """Q-Learning Visualizer

Red. prof. dr Samim Konjicija

Mašinsko učenje i inteligentno upravljanje

2025. godina"""

        messagebox.showinfo("About", about_text)


def main():
    root = tk.Tk()
    app = QLearningGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
