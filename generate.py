from utils import get_parser, load_model, get_dataloader, get_all_hidden_states, save_generations

def main(args):
    # Set up the model and data
    print("Loading model")
    model, tokenizer, model_type = load_model(args.model_name, args.cache_dir, args.parallelize, args.device)

    print("Loading dataloader")
    dataloader = get_dataloader(args.dataset_name, args.split, tokenizer, args.prompt_idx, batch_size=args.batch_size, 
                                num_examples=args.num_examples, model_type=model_type, use_decoder=args.use_decoder, device=args.device)

    # Get the hidden states and labels
    print("Generating hidden states")
    neg_hs, pos_hs, y = get_all_hidden_states(model, dataloader, layer=args.layer, all_layers=args.all_layers, 
                                              token_idx=args.token_idx, model_type=model_type, use_decoder=args.use_decoder)

    # Save the hidden states and labels
    print("Saving hidden states")
    save_generations(neg_hs, args, generation_type="negative_hidden_states")
    save_generations(pos_hs, args, generation_type="positive_hidden_states")
    save_generations(y, args, generation_type="labels")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
