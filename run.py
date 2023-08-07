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


def generate_and_evaluate(run_args, model, dataset, num_examples, prompt_idx):
    run_args.model_name = model
    run_args.batch_size = BATCH_SIZE
    dataset_name = dataset['dataset_name']
    dataset_dir = dataset['dataset_dir']
    run_args.dataset_name = dataset_name
    run_args.dataset_dir = dataset_dir
    run_args.num_examples = num_examples
    run_args.prompt_idx = prompt_idx
    print(f"Running generate using model '{model}' and prompt number {prompt_idx} with {num_examples} examples from "
          f"dataset {dataset_name} in directory {dataset_dir}")
    generate.main(run_args)
    print(f"Running evaluate using model '{model}' and prompt number {prompt_idx} with {num_examples} examples from "
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


def get_num_prompts(templates_file_path, dataset_name):
    with open(templates_file_path, "r") as f:
        templates = yaml.safe_load(f)
    # If we have templates for this dataset
    if templates['dataset'] == dataset_name:
        return len(templates['templates'])
    # No prompts found for this dataset
    raise Exception(f"No prompts found for {dataset_name}.")


def get_prompt_indices(dataset):
    dataset_prompt_indices = dataset['prompt_indices']
    # We have selected specific indices in the config file
    if len(dataset_prompt_indices) > 0:
        return dataset_prompt_indices
    # Use all prompts in the templates file
    else:
        num_prompts = get_num_prompts(template_file_path, dataset['dataset_name'])
        return range(num_prompts)


if __name__ == '__main__':
    yaml_config = parse_config()
    template_file_path = yaml_config['template_file_path']
    copy_templates_file(template_file_path)

    datasets = yaml_config['datasets']
    models = yaml_config['models']
    num_examples_l = yaml_config['num_training_examples']

    args_parser = get_parser()
    args = args_parser.parse_args()
    for model_name in models:
        for dataset_obj in datasets:
            for n_examples in num_examples_l:
                prompt_indices = get_prompt_indices(dataset_obj)
                for i in prompt_indices:
                    generate_and_evaluate(args, model_name, dataset_obj, n_examples, i)
