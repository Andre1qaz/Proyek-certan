import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import random
import math
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class NQueenSolver:
    """
    Kelas untuk menyelesaikan masalah N-Queen menggunakan berbagai algoritma Local Search
    """
    
    def __init__(self, n):
        self.n = n
        self.solution = None
        self.conflicts = 0
        self.iterations = 0
        self.execution_time = 0
        self.search_history = []
        
    def generate_random_state(self):
        """Menghasilkan state acak (satu ratu per kolom)"""
        return [random.randint(0, self.n - 1) for _ in range(self.n)]
    
    def calculate_conflicts(self, state):
        """
        Menghitung jumlah konflik (attacking pairs) dalam state
        Konflik terjadi jika dua ratu saling menyerang (diagonal)
        """
        conflicts = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                # Cek apakah di baris yang sama atau diagonal yang sama
                if state[i] == state[j]:  # Baris yang sama
                    conflicts += 1
                elif abs(state[i] - state[j]) == abs(i - j):  # Diagonal yang sama
                    conflicts += 1
        return conflicts
    
    def get_neighbors(self, state):
        """Menghasilkan semua neighbor states (tetangga)"""
        neighbors = []
        for col in range(self.n):
            for row in range(self.n):
                if row != state[col]:
                    neighbor = state.copy()
                    neighbor[col] = row
                    neighbors.append(neighbor)
        return neighbors
    
    def hill_climbing(self, max_iterations=1000):
        """
        Algoritma Hill-Climbing Search
        Mencari solusi dengan memilih neighbor terbaik di setiap iterasi
        """
        start_time = time.time()
        current = self.generate_random_state()
        current_conflicts = self.calculate_conflicts(current)
        
        self.search_history = [(0, current_conflicts)]
        self.iterations = 0
        
        while self.iterations < max_iterations:
            self.iterations += 1
            
            # Jika sudah menemukan solusi
            if current_conflicts == 0:
                self.solution = current
                self.conflicts = 0
                self.execution_time = time.time() - start_time
                return True, "Solusi ditemukan!"
            
            # Cari neighbor terbaik
            neighbors = self.get_neighbors(current)
            best_neighbor = None
            best_conflicts = current_conflicts
            
            for neighbor in neighbors:
                neighbor_conflicts = self.calculate_conflicts(neighbor)
                if neighbor_conflicts < best_conflicts:
                    best_neighbor = neighbor
                    best_conflicts = neighbor_conflicts
            
            # Jika tidak ada neighbor yang lebih baik (local maximum)
            if best_neighbor is None:
                self.solution = current
                self.conflicts = current_conflicts
                self.execution_time = time.time() - start_time
                return False, f"Terjebak di local maximum dengan {current_conflicts} konflik"
            
            current = best_neighbor
            current_conflicts = best_conflicts
            self.search_history.append((self.iterations, current_conflicts))
        
        self.solution = current
        self.conflicts = current_conflicts
        self.execution_time = time.time() - start_time
        return False, f"Maksimum iterasi tercapai dengan {current_conflicts} konflik"
    
    def random_restart_hill_climbing(self, max_restarts=100):
        """
        Algoritma Random-Restart Hill-Climbing
        Menjalankan hill-climbing berkali-kali dengan state awal berbeda
        """
        start_time = time.time()
        self.search_history = []
        total_iterations = 0
        
        for restart in range(max_restarts):
            current = self.generate_random_state()
            current_conflicts = self.calculate_conflicts(current)
            iterations = 0
            
            while iterations < 1000:
                iterations += 1
                total_iterations += 1
                
                if current_conflicts == 0:
                    self.solution = current
                    self.conflicts = 0
                    self.iterations = total_iterations
                    self.execution_time = time.time() - start_time
                    self.search_history.append((total_iterations, 0))
                    return True, f"Solusi ditemukan setelah {restart + 1} restart!"
                
                neighbors = self.get_neighbors(current)
                best_neighbor = None
                best_conflicts = current_conflicts
                
                for neighbor in neighbors:
                    neighbor_conflicts = self.calculate_conflicts(neighbor)
                    if neighbor_conflicts < best_conflicts:
                        best_neighbor = neighbor
                        best_conflicts = neighbor_conflicts
                
                if best_neighbor is None:
                    break
                
                current = best_neighbor
                current_conflicts = best_conflicts
                self.search_history.append((total_iterations, current_conflicts))
        
        self.solution = current
        self.conflicts = current_conflicts
        self.iterations = total_iterations
        self.execution_time = time.time() - start_time
        return False, f"Tidak menemukan solusi setelah {max_restarts} restart"
    
    def simulated_annealing(self, max_iterations=10000, initial_temp=100, cooling_rate=0.95):
        """
        Algoritma Simulated Annealing
        Mencari solusi dengan membolehkan move ke state yang lebih buruk dengan probabilitas tertentu
        """
        start_time = time.time()
        current = self.generate_random_state()
        current_conflicts = self.calculate_conflicts(current)
        
        temperature = initial_temp
        self.search_history = [(0, current_conflicts)]
        self.iterations = 0
        
        while self.iterations < max_iterations and temperature > 0.01:
            self.iterations += 1
            
            if current_conflicts == 0:
                self.solution = current
                self.conflicts = 0
                self.execution_time = time.time() - start_time
                return True, "Solusi ditemukan!"
            
            # Pilih neighbor acak
            col = random.randint(0, self.n - 1)
            row = random.randint(0, self.n - 1)
            neighbor = current.copy()
            neighbor[col] = row
            
            neighbor_conflicts = self.calculate_conflicts(neighbor)
            delta_e = current_conflicts - neighbor_conflicts
            
            # Terima neighbor jika lebih baik atau berdasarkan probabilitas
            if delta_e > 0 or random.random() < math.exp(delta_e / temperature):
                current = neighbor
                current_conflicts = neighbor_conflicts
            
            # Cooling schedule
            temperature *= cooling_rate
            
            if self.iterations % 100 == 0:
                self.search_history.append((self.iterations, current_conflicts))
        
        self.solution = current
        self.conflicts = current_conflicts
        self.execution_time = time.time() - start_time
        
        if current_conflicts == 0:
            return True, "Solusi ditemukan!"
        return False, f"Maksimum iterasi tercapai dengan {current_conflicts} konflik"


class NQueenGUI:
    """
    Kelas untuk GUI aplikasi N-Queen Solver
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("N-Queen Problem Solver - Local Search Algorithms")
        self.root.geometry("1400x900")
        
        self.solver = None
        self.results = {}
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup tampilan GUI"""
        # Frame utama
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="N-Queen Problem Solver", 
                                font=('Arial', 18, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Input Section
        input_frame = ttk.LabelFrame(main_frame, text="Input Parameter", padding="10")
        input_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(input_frame, text="Ukuran Board (N):").grid(row=0, column=0, padx=5)
        self.n_entry = ttk.Entry(input_frame, width=10)
        self.n_entry.insert(0, "8")
        self.n_entry.grid(row=0, column=1, padx=5)
        
        # Algorithm Selection
        algo_frame = ttk.LabelFrame(main_frame, text="Pilih Algoritma", padding="10")
        algo_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.algo_var = tk.StringVar(value="hill_climbing")
        
        ttk.Radiobutton(algo_frame, text="Hill-Climbing Search", 
                       variable=self.algo_var, value="hill_climbing").grid(row=0, column=0, padx=10)
        ttk.Radiobutton(algo_frame, text="Random-Restart Hill-Climbing", 
                       variable=self.algo_var, value="random_restart").grid(row=0, column=1, padx=10)
        ttk.Radiobutton(algo_frame, text="Simulated Annealing", 
                       variable=self.algo_var, value="simulated_annealing").grid(row=0, column=2, padx=10)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="Jalankan Algoritma", 
                  command=self.run_algorithm).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Bandingkan Semua Algoritma", 
                  command=self.compare_algorithms).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Clear", 
                  command=self.clear_results).grid(row=0, column=2, padx=5)
        
        # Results Section
        results_frame = ttk.LabelFrame(main_frame, text="Hasil Eksekusi", padding="10")
        results_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, width=80, height=15)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Visualization Section
        viz_frame = ttk.LabelFrame(main_frame, text="Visualisasi Board", padding="10")
        viz_frame.grid(row=4, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)
        
        self.canvas_frame = ttk.Frame(viz_frame)
        self.canvas_frame.grid(row=0, column=0)
        
        # Graph Section
        graph_frame = ttk.LabelFrame(main_frame, text="Grafik Konvergensi", padding="10")
        graph_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.graph_canvas_frame = ttk.Frame(graph_frame)
        self.graph_canvas_frame.grid(row=0, column=0)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(4, weight=1)
    
    def run_algorithm(self):
        """Menjalankan algoritma yang dipilih"""
        try:
            n = int(self.n_entry.get())
            if n < 4:
                messagebox.showerror("Error", "N harus minimal 4!")
                return
            
            self.solver = NQueenSolver(n)
            algo = self.algo_var.get()
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Menjalankan algoritma untuk N={n}...\n\n")
            self.root.update()
            
            if algo == "hill_climbing":
                success, message = self.solver.hill_climbing()
                algo_name = "Hill-Climbing Search"
            elif algo == "random_restart":
                success, message = self.solver.random_restart_hill_climbing()
                algo_name = "Random-Restart Hill-Climbing"
            else:
                success, message = self.solver.simulated_annealing()
                algo_name = "Simulated Annealing"
            
            self.display_results(algo_name, success, message)
            self.visualize_board()
            self.plot_convergence()
            
        except ValueError:
            messagebox.showerror("Error", "Masukkan nilai N yang valid!")
    
    def compare_algorithms(self):
        """Membandingkan semua algoritma"""
        try:
            n = int(self.n_entry.get())
            if n < 4:
                messagebox.showerror("Error", "N harus minimal 4!")
                return
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Membandingkan algoritma untuk N={n}...\n\n")
            self.root.update()
            
            self.results = {}
            
            # Hill-Climbing
            solver1 = NQueenSolver(n)
            success1, msg1 = solver1.hill_climbing()
            self.results['Hill-Climbing'] = {
                'success': success1,
                'message': msg1,
                'iterations': solver1.iterations,
                'conflicts': solver1.conflicts,
                'time': solver1.execution_time,
                'solution': solver1.solution
            }
            
            # Random-Restart Hill-Climbing
            solver2 = NQueenSolver(n)
            success2, msg2 = solver2.random_restart_hill_climbing()
            self.results['Random-Restart HC'] = {
                'success': success2,
                'message': msg2,
                'iterations': solver2.iterations,
                'conflicts': solver2.conflicts,
                'time': solver2.execution_time,
                'solution': solver2.solution
            }
            
            # Simulated Annealing
            solver3 = NQueenSolver(n)
            success3, msg3 = solver3.simulated_annealing()
            self.results['Simulated Annealing'] = {
                'success': success3,
                'message': msg3,
                'iterations': solver3.iterations,
                'conflicts': solver3.conflicts,
                'time': solver3.execution_time,
                'solution': solver3.solution
            }
            
            self.display_comparison()
            self.solver = solver2 if success2 else (solver3 if success3 else solver1)
            self.visualize_board()
            
        except ValueError:
            messagebox.showerror("Error", "Masukkan nilai N yang valid!")
    
    def display_results(self, algo_name, success, message):
        """Menampilkan hasil eksekusi algoritma"""
        self.results_text.insert(tk.END, f"{'='*60}\n")
        self.results_text.insert(tk.END, f"Algoritma: {algo_name}\n")
        self.results_text.insert(tk.END, f"{'='*60}\n\n")
        
        self.results_text.insert(tk.END, f"Status: {'SUKSES ✓' if success else 'GAGAL ✗'}\n")
        self.results_text.insert(tk.END, f"Pesan: {message}\n")
        self.results_text.insert(tk.END, f"Iterasi: {self.solver.iterations}\n")
        self.results_text.insert(tk.END, f"Konflik Akhir: {self.solver.conflicts}\n")
        self.results_text.insert(tk.END, f"Waktu Eksekusi: {self.solver.execution_time:.4f} detik\n\n")
        
        if self.solver.solution:
            self.results_text.insert(tk.END, f"Solusi (posisi kolom): {self.solver.solution}\n")
    
    def display_comparison(self):
        """Menampilkan perbandingan semua algoritma"""
        self.results_text.insert(tk.END, f"{'='*80}\n")
        self.results_text.insert(tk.END, f"PERBANDINGAN ALGORITMA\n")
        self.results_text.insert(tk.END, f"{'='*80}\n\n")
        
        header = f"{'Algoritma':<25} {'Status':<10} {'Iterasi':<12} {'Konflik':<10} {'Waktu (s)':<12}\n"
        self.results_text.insert(tk.END, header)
        self.results_text.insert(tk.END, f"{'-'*80}\n")
        
        for algo_name, result in self.results.items():
            status = "SUKSES ✓" if result['success'] else "GAGAL ✗"
            line = f"{algo_name:<25} {status:<10} {result['iterations']:<12} {result['conflicts']:<10} {result['time']:<12.4f}\n"
            self.results_text.insert(tk.END, line)
        
        self.results_text.insert(tk.END, f"\n{'='*80}\n")
        self.results_text.insert(tk.END, "ANALISIS:\n")
        self.results_text.insert(tk.END, f"{'='*80}\n\n")
        
        # Analisis perbandingan
        success_algos = [name for name, r in self.results.items() if r['success']]
        if success_algos:
            fastest = min(self.results.items(), key=lambda x: x[1]['time'] if x[1]['success'] else float('inf'))
            least_iter = min(self.results.items(), key=lambda x: x[1]['iterations'] if x[1]['success'] else float('inf'))
            
            self.results_text.insert(tk.END, f"✓ Algoritma tercepat: {fastest[0]} ({fastest[1]['time']:.4f} detik)\n")
            self.results_text.insert(tk.END, f"✓ Iterasi paling sedikit: {least_iter[0]} ({least_iter[1]['iterations']} iterasi)\n")
        else:
            self.results_text.insert(tk.END, "✗ Tidak ada algoritma yang berhasil menemukan solusi optimal.\n")
    
    def visualize_board(self):
        """Visualisasi papan catur dengan posisi queen"""
        if not self.solver or not self.solver.solution:
            return
        
        # Clear previous canvas
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        
        n = self.solver.n
        cell_size = min(400 // n, 50)
        
        canvas = tk.Canvas(self.canvas_frame, width=n*cell_size, height=n*cell_size)
        canvas.pack()
        
        # Draw board
        for row in range(n):
            for col in range(n):
                x1 = col * cell_size
                y1 = row * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                color = "#f0d9b5" if (row + col) % 2 == 0 else "#b58863"
                canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
        
        # Draw queens
        for col, row in enumerate(self.solver.solution):
            x = col * cell_size + cell_size // 2
            y = row * cell_size + cell_size // 2
            
            # Draw queen symbol
            canvas.create_text(x, y, text="♛", font=('Arial', int(cell_size * 0.6)), fill="red")
    
    def plot_convergence(self):
        """Plot grafik konvergensi algoritma"""
        if not self.solver or not self.solver.search_history:
            return
        
        # Clear previous plot
        for widget in self.graph_canvas_frame.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(12, 4))
        
        iterations = [x[0] for x in self.solver.search_history]
        conflicts = [x[1] for x in self.solver.search_history]
        
        ax.plot(iterations, conflicts, 'b-', linewidth=2)
        ax.set_xlabel('Iterasi', fontsize=12)
        ax.set_ylabel('Jumlah Konflik', fontsize=12)
        ax.set_title('Grafik Konvergensi Algoritma', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, self.graph_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
    
    def clear_results(self):
        """Clear semua hasil"""
        self.results_text.delete(1.0, tk.END)
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        for widget in self.graph_canvas_frame.winfo_children():
            widget.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = NQueenGUI(root)
    root.mainloop()