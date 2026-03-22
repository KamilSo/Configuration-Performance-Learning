import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


# Default configuration
DEFAULT_DATASET_ROOT = "Datasets"
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_REPEATS = 30
DEFAULT_TEST_SIZE = 0.30


# Core experiment functions
def detect_target_column(df: pd.DataFrame) -> str:
    possible_targets = [
        "time",
        "throughput",
        "runtime",
        "latency",
        "performance",
        "execution_time",
    ]

    lower_map = {col.lower(): col for col in df.columns}

    for col in possible_targets:
        if col.lower() in lower_map:
            return lower_map[col.lower()]

    print("Warning: Using last column as target")
    return df.columns[-1]


def validate_features(X: pd.DataFrame, dataset_name: str) -> None:
    non_numeric_columns = X.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
    if non_numeric_columns:
        raise ValueError(
            f"Dataset '{dataset_name}' contains non-numeric feature columns: {non_numeric_columns}"
        )


def mean_absolute_percentage_error_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    non_zero_mask = y_true != 0
    if non_zero_mask.sum() == 0:
        return np.nan

    mape = np.mean(
        np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
    ) * 100
    return mape


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error_safe(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
    }


def load_dataset(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, str]:
    df = pd.read_csv(csv_path)
    target_col = detect_target_column(df)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    validate_features(X, csv_path.stem)

    return X, y, target_col


def run_single_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    test_size: float,
) -> list[dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    run_results = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = evaluate_predictions(y_test, y_pred)
        metrics["model"] = model_name
        metrics["random_state"] = random_state
        run_results.append(metrics)

    return run_results


def summarise_results(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results_df.groupby(["dataset", "target", "model"])
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAPE_mean=("MAPE", "mean"),
            MAPE_std=("MAPE", "std"),
        )
        .reset_index()
    )
    return summary


def perform_wilcoxon_tests(results_df: pd.DataFrame) -> pd.DataFrame:
    test_results = []
    metrics = ["MAE", "RMSE", "MAPE"]

    for dataset in results_df["dataset"].unique():
        dataset_df = results_df[results_df["dataset"] == dataset]

        for metric in metrics:
            lr_values = dataset_df[dataset_df["model"] == "LinearRegression"][metric].values
            rf_values = dataset_df[dataset_df["model"] == "RandomForest"][metric].values

            if len(lr_values) != len(rf_values) or len(lr_values) == 0:
                continue

            if np.allclose(lr_values, rf_values):
                p_value = 1.0
                statistic = 0.0
            else:
                statistic, p_value = wilcoxon(lr_values, rf_values)

            significant = p_value < 0.05

            mean_lr = float(np.mean(lr_values))
            mean_rf = float(np.mean(rf_values))

            if mean_rf < mean_lr:
                better_model = "RandomForest"
            elif mean_lr < mean_rf:
                better_model = "LinearRegression"
            else:
                better_model = "Tie"

            test_results.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "significant_at_0_05": significant,
                    "better_mean_model": better_model,
                    "linear_mean": mean_lr,
                    "random_forest_mean": mean_rf,
                }
            )

    return pd.DataFrame(test_results)


def find_all_csv_files(root_folder: Path) -> list[Path]:
    if not root_folder.exists():
        return []

    excluded_folder_names = {"results", "__pycache__", ".idea"}
    excluded_file_names = {"detailed_results.csv", "summary_results.csv", "wilcoxon_results.csv"}

    csv_files = []

    for csv_path in root_folder.rglob("*.csv"):
        if any(part.lower() in excluded_folder_names for part in csv_path.parts):
            continue
        if csv_path.name.lower() in excluded_file_names:
            continue
        csv_files.append(csv_path)

    return sorted(csv_files)


# Graph plotting
def generate_dataset_boxplots(results_df: pd.DataFrame, plots_dir: Path) -> list[Path]:
    saved_files = []
    metrics = ["MAE", "RMSE", "MAPE"]

    for dataset in results_df["dataset"].unique():
        dataset_df = results_df[results_df["dataset"] == dataset]

        for metric in metrics:
            lr_values = dataset_df[dataset_df["model"] == "LinearRegression"][metric].values
            rf_values = dataset_df[dataset_df["model"] == "RandomForest"][metric].values

            if len(lr_values) == 0 or len(rf_values) == 0:
                continue

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.boxplot(
                [lr_values, rf_values],
                tick_labels=["LinearRegression", "RandomForest"]
            )
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} distribution for {dataset}")
            fig.tight_layout()

            output_file = plots_dir / f"{dataset}_{metric.lower()}_boxplot.png"
            fig.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close(fig)

            saved_files.append(output_file)

    return saved_files


def generate_metric_barplots(summary_df: pd.DataFrame, plots_dir: Path) -> list[Path]:
    saved_files = []
    metrics = ["MAE", "RMSE", "MAPE"]

    for metric in metrics:
        pivot_df = summary_df.pivot(index="dataset", columns="model", values=f"{metric}_mean")

        fig, ax = plt.subplots(figsize=(10, 6))
        pivot_df.plot(kind="bar", ax=ax)
        ax.set_ylabel(metric)
        ax.set_title(f"Average {metric} by dataset and model")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()

        output_file = plots_dir / f"{metric.lower()}_barplot.png"
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        saved_files.append(output_file)

    return saved_files


# GUI application
class ExperimentGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Configuration Performance Learning Tool")
        self.root.geometry("1000x760")

        self.dataset_root = tk.StringVar(value=DEFAULT_DATASET_ROOT)
        self.output_dir = tk.StringVar(value=DEFAULT_OUTPUT_DIR)
        self.repeats = tk.StringVar(value=str(DEFAULT_REPEATS))
        self.test_size = tk.StringVar(value=str(DEFAULT_TEST_SIZE))

        self.csv_files: list[Path] = []
        self.display_to_path: dict[str, Path] = {}

        self._build_ui()
        self.refresh_dataset_list()

    def _build_ui(self):
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill="x")

        ttk.Label(top_frame, text="Dataset root folder:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(top_frame, textvariable=self.dataset_root, width=60).grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(top_frame, text="Browse", command=self.browse_dataset_root).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(top_frame, text="Refresh CSV List", command=self.refresh_dataset_list).grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(top_frame, text="Output folder:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(top_frame, textvariable=self.output_dir, width=60).grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(top_frame, text="Browse", command=self.browse_output_dir).grid(row=1, column=2, padx=5, pady=5)

        ttk.Label(top_frame, text="Repeats:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(top_frame, textvariable=self.repeats, width=15).grid(row=2, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(top_frame, text="Test size:").grid(row=2, column=2, sticky="e", padx=5, pady=5)
        ttk.Entry(top_frame, textvariable=self.test_size, width=15).grid(row=2, column=3, sticky="w", padx=5, pady=5)

        top_frame.columnconfigure(1, weight=1)

        middle_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        middle_frame.pack(fill="both", expand=True)

        ttk.Label(middle_frame, text="Available CSV datasets (select one or more):").pack(anchor="w", pady=(0, 5))

        list_frame = ttk.Frame(middle_frame)
        list_frame.pack(fill="both", expand=True)

        self.dataset_listbox = tk.Listbox(
            list_frame,
            selectmode=tk.EXTENDED,
            width=120,
            height=20,
        )
        self.dataset_listbox.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.dataset_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.dataset_listbox.config(yscrollcommand=scrollbar.set)

        button_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        button_frame.pack(fill="x")

        ttk.Button(button_frame, text="Select All", command=self.select_all).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Clear Selection", command=self.clear_selection).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Run Experiment", command=self.run_experiment_thread).pack(side="right", padx=5)

        log_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        log_frame.pack(fill="both", expand=True)

        ttk.Label(log_frame, text="Log:").pack(anchor="w")

        self.log_text = tk.Text(log_frame, height=16, wrap="word")
        self.log_text.pack(fill="both", expand=True)

    def browse_dataset_root(self):
        folder = filedialog.askdirectory(title="Select dataset root folder")
        if folder:
            self.dataset_root.set(folder)
            self.refresh_dataset_list()

    def browse_output_dir(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.output_dir.set(folder)

    def refresh_dataset_list(self):
        root_folder = Path(self.dataset_root.get())
        self.csv_files = find_all_csv_files(root_folder)
        self.display_to_path.clear()

        self.dataset_listbox.delete(0, tk.END)

        for csv_path in self.csv_files:
            try:
                relative_path = csv_path.relative_to(root_folder)
            except ValueError:
                relative_path = csv_path

            display_name = str(relative_path)
            self.display_to_path[display_name] = csv_path
            self.dataset_listbox.insert(tk.END, display_name)

        self.log(f"Found {len(self.csv_files)} CSV files in '{root_folder}'.")

    def select_all(self):
        self.dataset_listbox.select_set(0, tk.END)

    def clear_selection(self):
        self.dataset_listbox.selection_clear(0, tk.END)

    def get_selected_files(self) -> list[Path]:
        selected_indices = self.dataset_listbox.curselection()
        selected_files = []

        for index in selected_indices:
            display_name = self.dataset_listbox.get(index)
            selected_files.append(self.display_to_path[display_name])

        return selected_files

    def log(self, message: str):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def run_experiment_thread(self):
        thread = threading.Thread(target=self.run_experiment, daemon=True)
        thread.start()

    def run_experiment(self):
        try:
            selected_files = self.get_selected_files()

            if not selected_files:
                messagebox.showwarning("No selection", "Please select at least one CSV dataset.")
                return

            try:
                n_repeats = int(self.repeats.get())
                test_size = float(self.test_size.get())
            except ValueError:
                messagebox.showerror("Invalid input", "Repeats must be an integer and test size must be a number.")
                return

            if n_repeats < 2:
                messagebox.showerror("Invalid input", "Repeats should be at least 2.")
                return

            if not (0 < test_size < 1):
                messagebox.showerror("Invalid input", "Test size must be between 0 and 1.")
                return

            output_dir = Path(self.output_dir.get())
            output_dir.mkdir(parents=True, exist_ok=True)

            plots_dir = output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            self.log("Starting experiment...")
            self.log(f"Selected {len(selected_files)} dataset(s).")
            self.log(f"Repeats: {n_repeats}")
            self.log(f"Test size: {test_size}")

            all_results = []

            for csv_path in selected_files:
                dataset_name = csv_path.stem

                self.log(f"\nProcessing dataset: {dataset_name}")
                self.log(f"File: {csv_path}")

                X, y, target_col = load_dataset(csv_path)

                self.log(f"Rows: {X.shape[0]}, Features: {X.shape[1]}, Target: {target_col}")

                for run_id in range(n_repeats):
                    run_output = run_single_experiment(
                        X=X,
                        y=y,
                        random_state=run_id,
                        test_size=test_size,
                    )

                    for row in run_output:
                        row["dataset"] = dataset_name
                        row["target"] = target_col
                        row["n_rows"] = X.shape[0]
                        row["n_features"] = X.shape[1]
                        row["run"] = run_id + 1
                        all_results.append(row)

            results_df = pd.DataFrame(all_results)
            summary_df = summarise_results(results_df)
            wilcoxon_df = perform_wilcoxon_tests(results_df)

            detailed_path = output_dir / "detailed_results.csv"
            summary_path = output_dir / "summary_results.csv"
            wilcoxon_path = output_dir / "wilcoxon_results.csv"

            results_df.to_csv(detailed_path, index=False)
            summary_df.to_csv(summary_path, index=False)
            wilcoxon_df.to_csv(wilcoxon_path, index=False)

            boxplot_files = generate_dataset_boxplots(results_df, plots_dir)
            barplot_files = generate_metric_barplots(summary_df, plots_dir)

            self.log("\nExperiment completed successfully.")
            self.log(f"Detailed results saved to: {detailed_path}")
            self.log(f"Summary results saved to: {summary_path}")
            self.log(f"Wilcoxon results saved to: {wilcoxon_path}")
            self.log(f"Generated {len(boxplot_files) + len(barplot_files)} plot(s) in: {plots_dir}")

            self.log("\nSummary:")
            self.log(summary_df.to_string(index=False))

            if not wilcoxon_df.empty:
                self.log("\nWilcoxon test results:")
                self.log(wilcoxon_df.to_string(index=False))

            messagebox.showinfo(
                "Done",
                f"Experiment finished.\n\nSaved:\n{detailed_path}\n{summary_path}\n{wilcoxon_path}\nPlots: {plots_dir}"
            )

        except Exception as e:
            self.log(f"\nError: {e}")
            messagebox.showerror("Error", str(e))

def main():
    root = tk.Tk()
    app = ExperimentGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()