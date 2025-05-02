#Mechine Learning Regression Project
#Yagegar-Imam University
#Dr S. Abolfazl Hosseini
#Gathered by Pooriya Babakhani using Chat-GPT & Qwen

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score

class RegressionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Regression Model GUI")
        self.df = None
        self.last_results = ""

        # Main layout using grid with 3 columns
        for i in range(4):
            self.master.grid_columnconfigure(i, weight=1)

        # Load File Button (centered)
        self.load_button = tk.Button(master, text="📂 Load Excel/CSV File", command=self.load_file, bg="#4CAF50", fg="white")
        self.load_button.grid(row=0, column=0, columnspan=4, pady=10)

        # X and Y Column Selection
        tk.Label(master, text="Select X columns:").grid(row=1, column=0, sticky='w', padx=10)
        self.x_listbox = tk.Listbox(master, selectmode='multiple', exportselection=False, width=30)
        self.x_listbox.grid(row=2, column=0, padx=10, pady=10)

        tk.Label(master, text="Select Y column:").grid(row=1, column=2, sticky='w')
        self.y_dropdown = tk.StringVar()
        self.y_menu = tk.OptionMenu(master, self.y_dropdown, "")
        self.y_menu.grid(row=2, column=2, padx=10, pady=10)

        # Models Section
        tk.Label(master, text="Models:").grid(row=3, column=0, sticky='w', padx=10)
        self.model_vars = {
            'Linear': tk.BooleanVar(value=True),
            'Poly': tk.BooleanVar(value=True)
        }
        for i, (name, var) in enumerate(self.model_vars.items()):
            tk.Checkbutton(master, text=name, variable=var).grid(row=4, column=0, sticky='w', padx=20+i*70)

        # Polynomial Degree
        tk.Label(master, text="Polynomial Degree:").grid(row=3, column=2, sticky='w', padx=10)
        self.degree_entry = tk.Entry(master, width=5)
        self.degree_entry.insert(0, "2")
        self.degree_entry.grid(row=4, column=2, sticky='w', padx=10)
        tk.Button(master, text="❓", command=lambda: messagebox.showinfo(
            "Help", "Polynomial Degree sets the complexity of polynomial models.\nHigher values may overfit.")).grid(
            row=4, column=2,columnspan=5)

        # Evaluation Method
        tk.Label(master, text="Evaluation:").grid(row=5, column=0, columnspan=4, pady=10)
        self.eval_method = tk.StringVar(value="Train/Test")
        #tk.Radiobutton(master, text="Train/Test Split", variable=self.eval_method, value="Train/Test").grid(row=6, column=1, sticky='w', padx=5)

        # Train % Entry
        tk.Label(master, text="Train/Test : -> Train %").grid(row=6, column=0, pady=10, columnspan=2)
        self.train_size_entry = tk.Entry(master, width=5)
        self.train_size_entry.insert(0, "70")
        self.train_size_entry.grid(row=6, column=1, pady=10)

        # Compare Degrees Button (centered)
        self.compare_button = tk.Button(master, text="📊 Compare Polynomial Degrees",
                                        command=self.compare_poly_degrees, bg="#2196F3", fg="white")
        self.compare_button.grid(row=7, column=0, columnspan=4, pady=10)

        # Run Models Button
        self.run_button = tk.Button(master, text="🚀 Run Models", command=self.run_models, bg="#FF9800", fg="white")
        self.run_button.grid(row=8, column=0, columnspan=4, pady=5)

        # Save Results Button
        self.save_button = tk.Button(master, text="💾 Save Results", command=self.save_results, bg="#9C27B0", fg="white")
        self.save_button.grid(row=9, column=0, columnspan=4, pady=5)

        # Status Bar
        self.status = tk.StringVar(value="Ready - Please load a file")
        self.status_label = tk.Label(master, textvariable=self.status, bd=1, relief=tk.SUNKEN)
        self.status_label.grid(row=10, column=0, columnspan=4, sticky='we')

        # Keyboard Shortcuts
        master.bind("<Control-l>", lambda e: self.load_file())
        master.bind("<Control-r>", lambda e: self.run_models())
        master.bind("<Control-d>", lambda e: self.compare_poly_degrees())
    def format_equation(self, coefs, intercept, feature_names):
        terms = []
        for coef, name in zip(coefs, feature_names):
            if abs(coef) < 0.001:
                continue
            terms.append(f"{coef:.2f}*{name}" if name != '1' else f"{coef:.2f}")
        if not terms:
            return "y = 0"
        equation = "y = " + " + ".join(terms)
        if intercept and abs(intercept) > 0.001:
            sign = "+" if intercept > 0 else "-"
            equation += f" {sign} {abs(intercept):.2f}"
        return equation.replace(" + -", " - ").replace("1.0*", "")

    def load_file(self):
        self.status.set("Loading file...")
        file_path = filedialog.askopenfilename(filetypes=[("Excel and CSV Files", "*.xlsx *.xls *.csv")])
        if file_path:
            try:
                if file_path.endswith(('.xlsx', '.xls')):
                    self.df = pd.read_excel(file_path)
                else:
                    self.df = pd.read_csv(file_path)
                self.df.dropna(inplace=True)
                numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
                if not numeric_cols:
                    raise ValueError("No numeric columns found after cleaning.")
                self.df = self.df[numeric_cols]
                columns = list(self.df.columns)
                self.x_listbox.delete(0, tk.END)
                for col in columns:
                    self.x_listbox.insert(tk.END, col)
                self.y_dropdown.set(columns[-1])
                menu = self.y_menu['menu']
                menu.delete(0, 'end')
                for col in columns:
                    menu.add_command(label=col, command=tk._setit(self.y_dropdown, col))
                self.status.set("File loaded successfully. Ready for analysis.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load or process file:\n{e}")
                self.status.set("Error loading file.")
        else:
            self.status.set("Ready")

    def compare_poly_degrees(self):
        if self.df is None:
            messagebox.showerror("Error", "No file loaded")
            return
        x_indices = self.x_listbox.curselection()
        x_cols = [self.x_listbox.get(i) for i in x_indices]
        y_col = self.y_dropdown.get()
        if not x_cols or not y_col:
            messagebox.showerror("Error", "Please select X and Y columns")
            return
        if len(x_cols) != 1:
            messagebox.showerror("Error", "Comparison works only with 1 feature.")
            return

        min_deg = simpledialog.askinteger("Minimum Degree", "Enter minimum degree:", minvalue=1, maxvalue=20)
        max_deg = simpledialog.askinteger("Maximum Degree", "Enter maximum degree:",
                                          minvalue=min_deg + 1 if min_deg else 2, maxvalue=30)
        if not min_deg or not max_deg:
            return

        X = self.df[[x_cols[0]]].values
        y = self.df[y_col].values

        plt.figure(figsize=(10, 6))
        x_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
        results = []

        for d in range(min_deg, max_deg + 1):
            poly = PolynomialFeatures(d)
            model = LinearRegression()
            X_poly = poly.fit_transform(X)
            model.fit(X_poly, y)
            y_pred = model.predict(X_poly)
            y_plot = model.predict(poly.transform(x_plot))
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rss = np.sum((y - y_pred) ** 2)
            plt.plot(x_plot, y_plot, label=f"Deg {d} (R²={r2:.2f})")
            results.append((d, r2, mse, rss))

        plt.scatter(X, y, color='black', alpha=0.3, label='Actual')
        plt.xlabel(x_cols[0])
        plt.ylabel(y_col)
        plt.title("Polynomial Degree Comparison")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        summary = "\n".join([f"Degree {d}: R²={r2:.3f}, MSE={mse:.1f}, RSS={rss:.1f}" for d, r2, mse, rss in results])
        messagebox.showinfo("Comparison Results", summary)
        self.status.set("Polynomial comparison completed.")

    def run_models(self):
        if self.df is None:
            messagebox.showerror("Error", "No file loaded")
            return
        x_indices = self.x_listbox.curselection()
        x_cols = [self.x_listbox.get(i) for i in x_indices]
        y_col = self.y_dropdown.get()
        if not x_cols or not y_col:
            messagebox.showerror("Error", "Please select X and Y columns")
            return

        X = self.df[x_cols].values
        y = self.df[y_col].values
        degree = int(self.degree_entry.get()) if self.degree_entry.get().isdigit() else 2
        is_single_feature = X.shape[1] == 1
        models_run = []
        eval_type = self.eval_method.get()

        for model_name, is_selected in self.model_vars.items():
            if not is_selected.get():
                continue
            if model_name == 'Linear':
                model = LinearRegression()
                X_input = X
                feature_names = x_cols
            else:
                poly = PolynomialFeatures(degree)
                X_input = poly.fit_transform(X)
                feature_names = poly.get_feature_names_out(x_cols)
                model = LinearRegression()

            if eval_type == "Train/Test":
                try:
                    train_percent = float(self.train_size_entry.get()) / 100
                    if not (0.05 <= train_percent <= 0.95):
                        raise ValueError("Train percentage must be between 5 and 95")
                    test_size = 1 - train_percent
                except Exception as e:
                    messagebox.showerror("Invalid Input", f"Invalid Train %: {e}")
                    return
                X_train, X_test, y_train, y_test = train_test_split(X_input, y, test_size=test_size, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                rss = np.sum((y_test - y_pred) ** 2)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
            else:
                scores = cross_val_score(model, X_input, y, scoring='r2', cv=5)
                model.fit(X_input, y)
                y_pred = model.predict(X_input)
                rss = np.sum((y - y_pred) ** 2)
                mse = mean_squared_error(y, y_pred)
                r2 = scores.mean()

            coefs = model.coef_
            intercept = model.intercept_
            equation = self.format_equation(coefs, intercept, feature_names)
            models_run.append((model_name, r2, mse, rss, equation))

            if is_single_feature and eval_type == "Train/Test":
                x_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
                if model_name == 'Linear':
                    y_plot = model.predict(x_plot)
                else:
                    x_plot_poly = poly.transform(x_plot)
                    y_plot = model.predict(x_plot_poly)
                plt.plot(x_plot, y_plot, label=f"{model_name} (R²={r2:.2f})")

            if len(x_cols) == 2 and eval_type == "Train/Test":
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure(figsize=(10, 7))
                ax = fig.add_subplot(111, projection='3d')
                x1 = self.df[x_cols[0]].values
                x2 = self.df[x_cols[1]].values
                ax.scatter(x1, x2, y, color='gray', alpha=0.5, label='Actual')

                for model_name, r2, _, _, _ in models_run:
                    grid_x, grid_y = np.meshgrid(
                        np.linspace(x1.min(), x1.max(), 30),
                        np.linspace(x2.min(), x2.max(), 30)
                    )
                    grid = np.c_[grid_x.ravel(), grid_y.ravel()]

                    if model_name == 'Linear':
                        temp_model = LinearRegression()
                        temp_model.fit(self.df[x_cols], y)
                        z = temp_model.predict(grid)
                    else:
                        poly = PolynomialFeatures(degree)
                        temp_model = LinearRegression()
                        X_poly = poly.fit_transform(self.df[x_cols])
                        temp_model.fit(X_poly, y)
                        z = temp_model.predict(poly.transform(grid))

                    ax.plot_surface(grid_x, grid_y, z.reshape(grid_x.shape), alpha=0.4, cmap='viridis')

                ax.set_xlabel(x_cols[0])
                ax.set_ylabel(x_cols[1])
                ax.set_zlabel(y_col)
                ax.set_title("3D Regression Plot")
                plt.tight_layout()
                plt.show()

            elif len(x_cols) <= 10 and eval_type == "Train/Test":
                fig, axes = plt.subplots(1, len(x_cols), figsize=(6 * len(x_cols), 5))
                if len(x_cols) == 1:
                    axes = [axes]
                for idx, ax in enumerate(axes):
                    x_feat = x_cols[idx]
                    X_plot = np.linspace(self.df[x_feat].min(), self.df[x_feat].max(), 300).reshape(-1, 1)

                    ax.scatter(self.df[x_feat], y, color='gray', alpha=0.5, label='Actual')
                    for model_name, r2, _, _, _ in models_run:
                        if model_name == 'Linear':
                            temp_model = LinearRegression()
                            temp_model.fit(self.df[[x_feat]], y)
                            y_plot = temp_model.predict(X_plot)
                        else:
                            poly = PolynomialFeatures(degree)
                            temp_model = LinearRegression()
                            X_poly = poly.fit_transform(self.df[[x_feat]])
                            temp_model.fit(X_poly, y)
                            y_plot = temp_model.predict(poly.transform(X_plot))

                        ax.plot(X_plot, y_plot, label=f"{model_name} (R²={r2:.2f})")

                    ax.set_title(f"Model vs {x_feat}")
                    ax.set_xlabel(x_feat)
                    ax.set_ylabel(y_col)
                    ax.legend()
                    ax.grid(True)

                plt.tight_layout()
                plt.show()

            elif len(x_cols) > 10:
                messagebox.showinfo("Info", "More than 10 X columns selected. Plotting skipped.")
            elif eval_type == "CV":
                messagebox.showinfo("Info", "Plotting skipped in Cross-Validation mode.")
            result_text = "\n\n".join([
                f"{name} Model:\nR² = {r2:.4f}\nMSE = {mse:.2f}\nRSS = {rss:.2f}\n{eq}"
                for name, r2, mse, rss, eq in models_run
            ])
            self.last_results = result_text
            messagebox.showinfo("Model Results", result_text)
            self.status.set("Analysis complete. Results ready.")

    def save_results(self):
        if not self.last_results:
            messagebox.showwarning("No Results", "No analysis results to save")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".txt",
                                              filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if filename:
            with open(filename, 'w') as f:
                f.write(self.last_results)
            messagebox.showinfo("Saved", f"Results saved to {filename}")

if __name__ == '__main__':
    root = tk.Tk()
    app = RegressionApp(root)
    root.mainloop()