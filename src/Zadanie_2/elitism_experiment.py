import numpy as np
import matplotlib.pyplot as plt
from evolution import evolution
from evaluation import evaluate

def run_elitism_experiment(pop_size=100, genome_size=400, max_iter=10000, 
                          mutation_rate=0.005, crossover_prob=0.8, trials=1):

    elite_values = range(1, 50)
    results = []
    
    for elite_size in elite_values:
        print(f"\nTesting elite_size = {elite_size}")
        trial_scores = []
        
        for t in range(trials):
            
            population = np.random.randint(0, 2, (pop_size, genome_size))
            
            best_ind, best_score = evolution(
                population=population,
                max_iter=max_iter,
                mutation_rate=mutation_rate,
                crossover_prob=crossover_prob,
                elite_size=elite_size
            )
            
            trial_scores.append(best_score)
            print(f"  Trial {t+1}/{trials}: score = {best_score}")
        
        avg_score = np.mean(trial_scores)
        results.append(avg_score)
        print(f"Average for elite_size={elite_size}: {avg_score:.2f}")
    
    return elite_values, results

def plot_elitism_results(elite_values, results):
    """Tworzy wykres wyników eksperymentu"""
    plt.figure(figsize=(10, 6))
    plt.plot(elite_values, results, 'bo-', linewidth=2, markersize=8)
    plt.title('Wpływ elityzmu na jakość rozwiązania', fontsize=14)
    plt.xlabel('Liczba elitarnych osobników', fontsize=12)
    plt.ylabel('Średni najlepszy fitness', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(elite_values)
    
    best_idx = np.argmax(results)
    plt.annotate(f'Najlepszy: {results[best_idx]:.2f}',
                 xy=(elite_values[best_idx], results[best_idx]),
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout()
    plt.savefig('elitism_impact.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    elite_values, results = run_elitism_experiment(
        pop_size=100,
        genome_size=400, 
        max_iter=10000,
        mutation_rate=0.005,
        crossover_prob=0.8,
        trials=1
    )
    
    plot_elitism_results(elite_values, results)