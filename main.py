from argparse import ArgumentParser
from dataset import resolve_datasets
from tagger import resolve_taggers
from evaluator import evaluate_treebank_dataset, test_treebank_dataset


def run_program(type_, data_dir, tagger_type, test_string=""):
    try:
        # get taggers
        taggers = resolve_taggers(tagger_type)
        if type_ == "eval":
            conf_matrix = dict()
            for dataset in resolve_datasets(data_dir):
                conf_m = evaluate_treebank_dataset(dataset, taggers)
                conf_matrix.update(conf_m)
            # TODO do something with confusion matrix
        elif type_ == "test":
            for dataset in resolve_datasets(data_dir):
                result = test_treebank_dataset(dataset, taggers, test_string)
            # TODO do something with the result
        else:
            raise ValueError(f"Cannot process the run type [{type_}]")
    except Exception as e:
        print("\n**Error!!!**\n")
        print("--------")
        print(e)
        print("--------")


if __name__ == "__main__":
    parser = ArgumentParser(description="||| POSTagging for Treebank Datasets (Practical 1) ||| ")
    parser.add_argument("Run type", metavar="run_type", type=str, help="One of [test] or [eval]")
    parser.add_argument("-d", "--data_dir", type=str,
                        help="One of [all] [directory containing treebank file]", default="all")
    parser.add_argument("-t", "--tagger", type=str, help="One of [eager] [viterbi] [local_decoding] [all]", default="all")
    parser.add_argument("-s", "--test_string", type=str, help="text to generate tags for", default="")
    args = parser.parse_args()
    run_program(args.run_type, args.data_dir, args.tagger, args.test_string)
