import os
import yaml
import argparse
import site
import shutil
from utils import get_parser
import generate
import evaluate

BATCH_SIZE = 6


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--device", type=str, default="cpu", help="Device on which to run the script")
    parsed_args = parser.parse_args()

    with open(parsed_args.config_file_path, "r") as f:
        config = yaml.safe_load(f)

    return config, parsed_args.device


def generate_and_evaluate(run_parser, model, dataset, num_examples, prompt_idx,
                          no_data_balance, device, split, threshold):
    run_args = run_parser.parse_args()
    run_args.model_name = model
    run_args.batch_size = BATCH_SIZE
    dataset_name = dataset['dataset_name']
    dataset_dir = dataset['dataset_dir']
    run_args.dataset_name = dataset_name
    run_args.dataset_dir = dataset_dir
    run_args.num_examples = num_examples
    run_args.prompt_idx = prompt_idx
    run_args.no_data_balance = no_data_balance
    run_args.device = device
    run_args.split = split
    run_args.threshold = threshold
    print("-" * 200)
    args_string = f"Model: '{model}'\n" \
                  f"Prompt Number: '{prompt_idx}'\n" \
                  f"Number of Examples: '{num_examples}'\n" \
                  f"Dataset Name: '{dataset_name}'\n" \
                  f"Dataset Directory: '{dataset_dir}'\n" \
                  f"No Data Balance: '{no_data_balance}'\n" \
                  f"Toxicity Threshold: '{threshold}'"

    print(f"Running generate with the following arguments:\n {args_string}")
    generate.main(run_args)
    print(f"Running evaluate with the following arguments:\n {args_string}")
    evaluate.main(run_parser, run_args)


def copy_templates_file(templates_file_path, dataset_name):
    site_packages_path = site.getsitepackages()[0]
    template_path_suffix = f"promptsource/templates/{dataset_name}"
    templates_file_target_path = f"{site_packages_path}/{template_path_suffix}/templates.yaml"
    print(f"Copying templates file {templates_file_path} to {templates_file_target_path}")
    shutil.os.system(f"sudo cp {templates_file_path} {templates_file_target_path}")
    print("Successfully copied templates file")


def get_num_prompts(templates_file_path, dataset_name):
    with open(templates_file_path, "r") as f:
        templates = yaml.full_load(f)
    # If we have templates for this dataset
    if templates['dataset'] == dataset_name:
        return len(templates['templates'])
    # No prompts found for this dataset
    raise Exception(f"No prompts found for {dataset_name}.")


def get_prompt_indices(dataset, template_path):
    dataset_prompt_indices = dataset.get('prompt_indices')
    # Use all prompts in the templates file
    if dataset_prompt_indices is None or len(dataset_prompt_indices) == 0:
        num_prompts = get_num_prompts(template_path, dataset['dataset_name'])
        return range(num_prompts)
    # We have selected specific indices in the config file
    else:
        return dataset_prompt_indices


if __name__ == '__main__':
    # Set TOKENIZERS_PARALLELISM in order to avoid huggingface warnings
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    yaml_config, dvc = parse_config()

    datasets = yaml_config['datasets']
    models = yaml_config['models']
    num_examples_l = yaml_config['num_training_examples']
    no_data_balance_l = yaml_config['no_data_balance']

    for model_name in models:
        for dataset_obj in datasets:
            template_file_path = dataset_obj.get('template_file_path')
            data_split = dataset_obj['data_split']
            if template_file_path is not None:
                copy_templates_file(template_file_path, dataset_obj['dataset_name'])
            for n_examples in num_examples_l:
                prompt_indices = get_prompt_indices(dataset_obj, template_file_path)
                for i in prompt_indices:
                    for should_data_balance in no_data_balance_l:
                        # Iterate over thresholds if we've set them in the config
                        for toxic_threshold in (dataset_obj.get('thresholds') or []):
                            args_parser = get_parser()
                            generate_and_evaluate(args_parser, model_name, dataset_obj, n_examples, i,
                                                  should_data_balance, dvc, data_split, toxic_threshold)
    print("Finished running generate and evaluate with all model configurations")
