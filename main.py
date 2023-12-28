import tkinter as tk
from svmGeneticalgo import best_solution as best_solution_svmGeneticalgo
from randomforestGeneticalgo import best_solution as best_solution_rainforestGeneticalgo
from gradientboosterGeneticalgo import best_solution as best_solution_gradientboosterGeneticalgo
from svmHillclimbing import best_solution as best_solution_svmHillClimbing

def on_select(val):
    if val == "SVM-genetic algorithm":
        solution = best_solution_svmGeneticalgo
    elif val == "SVM-Hill climbing algorithm":
        solution = best_solution_svmHillClimbing
    elif val == "Rain forest - Genetic Algorithm":
        solution = best_solution_rainforestGeneticalgo
    elif val == "Gradient booster - genetic algorithm":
        solution = best_solution_gradientboosterGeneticalgo
    else:
        solution = "Unknown algorithm"

    selected_option.set(f"Selected: {val}\nBest Solution: {solution}")

# Initialize main window
root = tk.Tk()
root.title("Dropdown Example")
root.geometry("300x200")

selected_option = tk.StringVar()

options = ["SVM-genetic algorithm", "SVM-Hill climbing algorithm", "Rain forest - Genetic Algorithm", "Gradient booster - genetic algorithm"]
dropdown = tk.OptionMenu(root, selected_option, *options, command=on_select)
dropdown.pack()

output_label = tk.Label(root, textvariable=selected_option)
output_label.pack()

root.mainloop()
