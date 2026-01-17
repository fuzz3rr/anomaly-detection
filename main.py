import argparse
import glob
import json
import os
import sys
import warnings


class TeeOutput:
    """Duplicate stdout to both console and a file."""
    def __init__(self, filepath):
        self.file = open(filepath, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    
    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.file.flush()
    
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()
        sys.stdout = self.stdout

import pandas as pd
from sklearn.model_selection import train_test_split

from classifiers import Classifier
from data_utils import (
    load_data,
    preprocess_data,
    prepare_binary_labels,
    prepare_multiclass_labels,
    get_binary_target_names,
    scale_features,
)
from plots import (
    plot_confusion_matrices,
    plot_feature_importance,
    plot_class_distribution,
    plot_multiclass_only_confusion_matrices,
    plot_binary_only_confusion_matrices,
    plot_model_comparison,
)

warnings.filterwarnings("ignore")

# Default paths
DEFAULT_DATASETS_CONFIG = "datasets/datasets.json"
DEFAULT_MODELS_DIR = "models"


def print_header(text, char="="):
    print("\n" + char * 80)
    print(text)
    print(char * 80)


def print_subheader(text):
    print_header(text, char="-")


def print_results(model_name, classification_type, results):
    print(f"\n{model_name} ({classification_type}):")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1-score: {results['f1_score']:.4f}")
    print("\nClassification Report:")
    print(results["report"])


def run_binary_classification(X_train, X_test, y_train, y_test, target_names, model_types, model_params):
    print("\n[BINARY CLASSIFICATION]")
    results = []

    for model_type in model_types:
        params = model_params.get(model_type, {})
        classifier = Classifier(model_type=model_type, num_classes=2, model_params=params)
        classifier.train(X_train, y_train)
        eval_results = classifier.evaluate(X_test, y_test, target_names=target_names)
        print_results(classifier.name, "Binary Classification", eval_results)

        results.append(
            {
                "model_name": classifier.name,
                "model_type": model_type,
                "classification_type": "Binary",
                "accuracy": eval_results["accuracy"],
                "f1_score": eval_results["f1_score"],
                "predictions": eval_results["predictions"],
            }
        )

    return results


def run_multiclass_classification(X_train, X_test, y_train, y_test, class_names, model_types, model_params):
    print("\n[MULTICLASS CLASSIFICATION]")
    results = []
    models_importance = []

    for model_type in model_types:
        params = model_params.get(model_type, {})
        classifier = Classifier(model_type=model_type, num_classes=len(class_names), model_params=params)
        classifier.train(X_train, y_train)
        eval_results = classifier.evaluate(X_test, y_test, target_names=class_names)
        print_results(classifier.name, "Multiclass Classification", eval_results)

        results.append(
            {
                "model_name": classifier.name,
                "model_type": model_type,
                "classification_type": "Multiclass",
                "accuracy": eval_results["accuracy"],
                "f1_score": eval_results["f1_score"],
                "predictions": eval_results["predictions"],
            }
        )

        importances = classifier.get_feature_importances()
        models_importance.append({"name": classifier.name, "importances": importances})

    return results, models_importance


def save_results(binary_results, multiclass_results, output_path, dataset_name, config_name):
    all_results = binary_results + multiclass_results

    results_df = pd.DataFrame(
        [
            {
                "Model Config": config_name,
                "Dataset": dataset_name,
                "Model": r["model_name"],
                "Classification Type": r["classification_type"],
                "Accuracy": r["accuracy"],
                "F1-score": r["f1_score"],
            }
            for r in all_results
        ]
    )

    print_subheader("RESULTS SUMMARY")
    print(results_df.to_string(index=False))

    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")

    return results_df


def check_prerequisites(file_paths, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    missing_files = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print(f"✗ Error: Data files not found:")
        for f in missing_files:
            print(f"    - {f}")
        return False

    return True


def list_datasets(datasets_config):
    print("\nAvailable datasets:")
    for key, dataset in datasets_config.items():
        multiclass_only = dataset.get("multiclass_only", False)
        binary_only = dataset.get("binary_only", False)

        if multiclass_only:
            classification_type = "multiclass only"
        elif binary_only:
            classification_type = "binary only"
        else:
            classification_type = "binary + multiclass"

        print(f"  - {key}: {dataset['name']} ({classification_type})")


def list_model_configs(models_dir):
    print(f"\nAvailable model configurations in '{models_dir}/':")
    config_files = glob.glob(os.path.join(models_dir, "*.json"))

    if not config_files:
        print("  No configuration files found!")
        return []

    configs = []
    for config_file in sorted(config_files):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            name = config.get("name", os.path.basename(config_file))
            desc = config.get("description", "No description")
            print(f"  - {os.path.basename(config_file)}: {name}")
            print(f"      {desc}")
            configs.append(config_file)
        except Exception as e:
            print(f"  - {os.path.basename(config_file)}: Error loading ({e})")

    return configs


def load_model_configs(models_dir, specific_config=None):
    """Load model configurations from directory or specific file."""
    if specific_config:
        # Load specific config(s)
        configs = []
        for config_name in specific_config:
            # Try exact path first
            if os.path.exists(config_name):
                config_path = config_name
            else:
                # Try in models directory
                config_path = os.path.join(models_dir, config_name)
                if not os.path.exists(config_path):
                    # Try with .json extension
                    config_path = os.path.join(models_dir, f"model_{config_name}.json")

            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                config['_filepath'] = config_path
                configs.append(config)
            else:
                print(f"⚠️  Warning: Config '{config_name}' not found, skipping.")
        return configs
    else:
        # Load all configs from directory
        config_files = glob.glob(os.path.join(models_dir, "*.json"))
        configs = []
        for config_file in sorted(config_files):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                config['_filepath'] = config_file
                configs.append(config)
            except Exception as e:
                print(f"⚠️  Warning: Could not load {config_file}: {e}")
        return configs


def run_single_analysis(dataset_config, model_config, file_paths, output_dir, model_types, verbose=True):
    """Run analysis for a single dataset + model configuration combination."""

    config_name = model_config.get("name", "Unknown")
    dataset_name = dataset_config["name"]
    preprocessing_config = model_config.get("preprocessing", {})
    model_params = model_config.get("models", {})
    label_column = dataset_config.get("label_column", "Label")
    is_multiclass_only = dataset_config.get("multiclass_only", False)
    is_binary_only = dataset_config.get("binary_only", False)

    if verbose:
        print(f"\n📊 Loading data...")
    data = load_data(file_paths, dataset_config)

    if verbose:
        print(f"\n🔧 Preprocessing data...")
    data = preprocess_data(data, dataset_config, preprocessing_config)

    if verbose:
        print(f"Features: {data.shape[1] - 1}")
        print("\nClass distribution:")
        for label, count in data[label_column].value_counts().items():
            print(f"  {label}: {count} ({count / len(data) * 100:.1f}%)")

    # Validate number of classes
    n_classes = data[label_column].nunique()
    if n_classes < 2:
        print(f"\n⚠️  WARNING: Only {n_classes} class found. Skipping this configuration.")
        return None, None

    X = data.drop(label_column, axis=1)
    y = data[label_column]

    binary_results = []
    y_test_bin = None

    # Binary classification
    if not is_multiclass_only:
        if verbose:
            print(f"\n🎯 Running binary classification...")

        positive_class = dataset_config.get("binary_positive_class", "BENIGN")
        binary_target_names = get_binary_target_names(dataset_config)

        y_binary = prepare_binary_labels(y, positive_class)
        X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
            X, y_binary,
            test_size=preprocessing_config.get("test_size", 0.3),
            random_state=preprocessing_config.get("random_state", 42),
            stratify=y_binary
        )
        X_train_bin_scaled, X_test_bin_scaled = scale_features(X_train_bin, X_test_bin)

        binary_results = run_binary_classification(
            X_train_bin_scaled, X_test_bin_scaled,
            y_train_bin, y_test_bin,
            binary_target_names,
            model_types, model_params
        )

    # Multiclass classification (skip if binary_only)
    multiclass_results = []
    models_importance = []
    y_test_multi = None
    label_encoder = None

    if not is_binary_only:
        if verbose:
            print(f"\n🎯 Running multiclass classification...")

        y_multi, label_encoder = prepare_multiclass_labels(y)
        X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
            X, y_multi,
            test_size=preprocessing_config.get("test_size", 0.3),
            random_state=preprocessing_config.get("random_state", 42),
            stratify=y_multi
        )
        X_train_multi_scaled, X_test_multi_scaled = scale_features(X_train_multi, X_test_multi)

        multiclass_results, models_importance = run_multiclass_classification(
            X_train_multi_scaled,
            X_test_multi_scaled,
            y_train_multi,
            y_test_multi,
            label_encoder.classes_,
            model_types,
            model_params,
        )

    # Generate visualizations
    if verbose:
        print(f"\n📈 Generating visualizations...")

    config_slug = config_name.lower().replace(" ", "_")

    if is_binary_only:
        # Binary only - plot binary confusion matrices
        plot_binary_only_confusion_matrices(
            binary_results,
            y_test_bin,
            get_binary_target_names(dataset_config),
            f"{output_dir}/confusion_matrices_{config_slug}.png"
        )
    elif is_multiclass_only:
        # Multiclass only
        plot_multiclass_only_confusion_matrices(
            multiclass_results,
            y_test_multi,
            label_encoder.classes_,
            f"{output_dir}/confusion_matrices_{config_slug}.png"
        )
    else:
        # Both binary and multiclass
        viz_results = []
        for bin_res, multi_res in zip(binary_results, multiclass_results):
            viz_results.append(
                {
                    "model_name": bin_res["model_name"],
                    "y_test_bin": y_test_bin,
                    "y_pred_bin": bin_res["predictions"],
                    "y_test_multi": y_test_multi,
                    "y_pred_multi": multi_res["predictions"],
                    "class_names": label_encoder.classes_,
                    "binary_labels": get_binary_target_names(dataset_config),
                }
            )
        plot_confusion_matrices(viz_results, f"{output_dir}/confusion_matrices_{config_slug}.png")

    # Feature importance (only if we have multiclass results with tree-based models)
    if models_importance:
        plot_feature_importance(
            models_importance, X.columns, f"{output_dir}/feature_importance_{config_slug}.png"
        )

    # Save results
    results_df = save_results(
        binary_results,
        multiclass_results,
        f"{output_dir}/results_{config_slug}.csv",
        dataset_name,
        config_name,
    )

    return results_df, y


def main():
    parser = argparse.ArgumentParser(
        description="Universal Network Traffic Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available options
  python main.py --list-datasets
  python main.py --list-models

  # Run with all model configs
  python main.py -d ddos                          # All configs on DDoS dataset
  python main.py -d darknet_tor -s                # All configs on smaller dataset

  # Run with specific model config(s)
  python main.py -d ddos -c baseline              # Single config
  python main.py -d ddos -c baseline fast_test    # Multiple configs
  python main.py -d ddos -c models/baseline.json  # Full path

  # Additional options
  python main.py -d ddos -m rf xgb                # Only specific models
  python main.py -d ddos --models-dir custom/     # Custom configs directory
        """
    )

    # Dataset options
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        help='Dataset to analyze (use --list-datasets to see options)'
    )
    parser.add_argument(
        '--smaller', '-s',
        action='store_true',
        help='Use smaller dataset for quick testing'
    )
    parser.add_argument(
        '--datasets-config',
        type=str,
        default=DEFAULT_DATASETS_CONFIG,
        help=f'Path to datasets configuration (default: {DEFAULT_DATASETS_CONFIG})'
    )

    # Model config options
    parser.add_argument(
        '--config', '-c',
        nargs='+',
        help='Specific model config(s) to use. If not specified, runs all configs from models directory'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default=DEFAULT_MODELS_DIR,
        help=f'Directory with model configurations (default: {DEFAULT_MODELS_DIR})'
    )
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        default=['rf', 'xgb', 'svm'],
        choices=['rf', 'xgb', 'svm'],
        help='ML models to use (default: rf xgb svm)'
    )

    # List options
    parser.add_argument(
        '--list-datasets', '-ld',
        action='store_true',
        help='List available datasets and exit'
    )
    parser.add_argument(
        '--list-models', '-lm',
        action='store_true',
        help='List available model configurations and exit'
    )

    # Output options
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory (default: outputs/<dataset>)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce output verbosity'
    )
    parser.add_argument(
        '--log-file', '-lf',
        type=str,
        help='Save all output to a log file (also prints to console)'
    )

    args = parser.parse_args()

    # Setup log file if specified
    tee = None
    if args.log_file:
        log_dir = os.path.dirname(args.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        tee = TeeOutput(args.log_file)
        sys.stdout = tee

    # Load datasets configuration
    if not os.path.exists(args.datasets_config):
        print(f"Error: Datasets config not found at '{args.datasets_config}'")
        if tee:
            tee.close()
        sys.exit(1)

    with open(args.datasets_config, 'r') as f:
        datasets_config = json.load(f)

    # Handle list commands
    if args.list_datasets:
        list_datasets(datasets_config)
        sys.exit(0)

    if args.list_models:
        list_model_configs(args.models_dir)
        sys.exit(0)

    # Validate dataset selection
    if not args.dataset:
        print("Error: Please specify a dataset with --dataset")
        list_datasets(datasets_config)
        sys.exit(1)

    if args.dataset not in datasets_config:
        print(f"Error: Unknown dataset '{args.dataset}'")
        list_datasets(datasets_config)
        sys.exit(1)

    # Load dataset config
    dataset_config = datasets_config[args.dataset]
    dataset_name = dataset_config["name"]

    # Determine file paths
    if args.smaller:
        file_paths = dataset_config.get("files_smaller", dataset_config["files"])
    else:
        file_paths = dataset_config["files"]

    # Setup output directory
    output_dir = args.output_dir or f"outputs/{args.dataset}"

    if not check_prerequisites(file_paths, output_dir):
        sys.exit(1)

    # Load model configurations
    model_configs = load_model_configs(args.models_dir, args.config)

    if not model_configs:
        print(f"Error: No model configurations found!")
        print(f"Please add configuration files to '{args.models_dir}/' or specify with --config")
        sys.exit(1)

    print_header(f"NETWORK TRAFFIC CLASSIFICATION: {dataset_name.upper()}")
    print(f"\n📁 Dataset: {args.dataset}")
    print(f"📁 Output: {output_dir}")
    print(f"🔬 Models: {', '.join(args.models)}")
    print(f"⚙️  Configurations to run: {len(model_configs)}")

    for cfg in model_configs:
        print(f"    - {cfg.get('name', 'Unknown')}")

    # Run analysis for each model configuration
    all_results = []
    class_distribution_saved = False

    for i, model_config in enumerate(model_configs, 1):
        config_name = model_config.get("name", f"Config {i}")
        print_header(f"RUNNING: {config_name} ({i}/{len(model_configs)})")

        try:
            results_df, y = run_single_analysis(
                dataset_config=dataset_config,
                model_config=model_config,
                file_paths=file_paths,
                output_dir=output_dir,
                model_types=args.models,
                verbose=not args.quiet
            )

            if results_df is not None:
                all_results.append(results_df)

                # Save class distribution only once
                if not class_distribution_saved and y is not None:
                    plot_class_distribution(y, f"{output_dir}/class_distribution.png")
                    class_distribution_saved = True

        except Exception as e:
            print(f"❌ Error running {config_name}: {e}")
            if not args.quiet:
                import traceback
                traceback.print_exc()

    # Combine and save all results
    if all_results:
        print_header("FINAL SUMMARY")

        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv(f"{output_dir}/all_results.csv", index=False)

        print("\n📊 Combined Results:")
        print(combined_results.to_string(index=False))

        # Generate comparison plot
        if len(model_configs) > 1:
            plot_model_comparison(combined_results, f"{output_dir}/model_comparison.png")

        print(f"\n✓ All results saved to: {output_dir}/all_results.csv")

    print_header("ANALYSIS COMPLETED")
    print(f"\n📁 Output files saved to: {output_dir}/")

    # Close log file if used
    if tee:
        print(f"📄 Log saved to: {args.log_file}")
        tee.close()


if __name__ == "__main__":
    main()