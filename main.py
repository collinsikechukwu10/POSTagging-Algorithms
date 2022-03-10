import traceback
from argparse import ArgumentParser
from dataset import resolve_datasets
from tagger import resolve_taggers
from evaluator import evaluate_treebank_dataset, test_treebank_dataset
import json


def run_program(type_, dataset_code, tagger_type, test_string=""):
    """
    Main entrypoint of the program. Runs the test or evaluation tool.
    :param type_: Run type
    :param dataset_code: Dataset language code
    :param tagger_type: Tagger type
    :param test_string: String to test if type_ is `test`
    """
    try:
        # get taggers
        taggers = resolve_taggers(tagger_type)
        if type_ == "eval":
            conf_matrix = dict()
            for dataset in resolve_datasets(dataset_code):
                print(f"Evaluating Treebank test dataset: {dataset.name()}")
                conf_m = evaluate_treebank_dataset(dataset, taggers)
                conf_matrix.update(conf_m)
        elif type_ == "test":
            for dataset in resolve_datasets(dataset_code):
                result = test_treebank_dataset(dataset, taggers, test_string)
                # At the moment we can just store the
                print(json.dumps(result, indent=4))
        else:
            raise ValueError(f"Cannot process the run type [{type_}]")
    except Exception as e:
        print("\n**Error!!!**\n")
        print("--------")
        print(traceback.format_exc())
        print("--------")


if __name__ == "__main__":
    parser = ArgumentParser(description="||| POSTagging for Treebank Datasets (Practical 1) ||| ")
    parser.add_argument("-r", "--run_type", type=str, help="One of [test] or [eval]", required=True)
    parser.add_argument("-d", "--dataset_code", type=str,
                        help="One of [all] [...language code]", default="all")
    parser.add_argument("-t", "--tagger", type=str, help="One of [eager] [viterbi] [local_decoding] [all]",
                        default="all")
    parser.add_argument("-s", "--test_string", type=str, help="text to generate tags for", default="")
    args = parser.parse_args()
    run_program(args.run_type, args.dataset_code, args.tagger, args.test_string)
