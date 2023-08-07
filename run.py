import os.path

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
    parsed_args = parser.parse_args()

    with open(parsed_args.config_file_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def generate_and_evaluate(run_args, model, dataset, num_examples):
    run_args.model_name = model
    run_args.batch_size = BATCH_SIZE
    dataset_name = dataset['dataset_name']
    dataset_dir = dataset['dataset_dir']
    run_args.dataset_name = dataset_name
    run_args.dataset_dir = dataset_dir
    run_args.num_examples = num_examples
    print(f"Running generate for model {model} with {num_examples} examples from "
          f"dataset {dataset_name} in {dataset_dir}")
    generate.main(run_args)
    print(f"Running evaluate for model {model} with {num_examples} examples from "
          f"dataset {dataset_name} in {dataset_dir}")
    evaluate.main(get_parser())


def copy_templates_file(templates_file_path):
    templates_file_name = os.path.basename(templates_file_path)
    site_packages_path = site.getsitepackages()[0]
    template_path_suffix = "promptsource/templates"
    templates_file_target_path = f"{site_packages_path}/{template_path_suffix}/{templates_file_name}"
    print(f"Copying templates file {templates_file_path} to {templates_file_target_path}")
    shutil.copyfile(templates_file_path, templates_file_target_path)
    print("Successfully copied templates file")


if __name__ == '__main__':
    yaml_config = parse_config()
    template_file_path = yaml_config['template_file_path']
    copy_templates_file(template_file_path)

    datasets = yaml_config['datasets']
    models = yaml_config['models']
    num_examples_l = yaml_config['num_examples']

    args_parser = get_parser()
    args = args_parser.parse_args()
    for model_name in models:
        for dataset_obj in datasets:
            for n_examples in num_examples_l:
                generate_and_evaluate(args, model_name, dataset_obj, n_examples)
