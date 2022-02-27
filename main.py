import os
from argparse import ArgumentParser
from probability import BiGramProbabilityMatrix
from dataset import prepare_datasets, TreeBankDataset
from tagger import EagerTagger, VibertiPOSTagger, MostProbablePOSTagger
from evaluator import POSEvaluator

TREE_BANKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "treebanks")


def run_evaluation(dataset: TreeBankDataset):
    train_dataset = dataset.train_data()
    test_dataset = dataset.test_data()
    # get transition probability using training sample
    transition_probability = BiGramProbabilityMatrix(
        unique_prior_tokens=train_dataset.unique_tags(),
        unique_target_tokens=train_dataset.unique_tags(),
        ngrams_list=train_dataset.transitions())
    # get emission probability
    emission_probability = BiGramProbabilityMatrix(
        unique_prior_tokens=train_dataset.unique_tags(),
        unique_target_tokens=train_dataset.unique_vocabulary(),
        ngrams_list=train_dataset.emmisions())
    eager_tagger = EagerTagger(emission_probability, transition_probability)
    viberti_tagger = VibertiPOSTagger(emission_probability, transition_probability)
    mp_tagger = MostProbablePOSTagger(emission_probability, transition_probability)
    confusion_matrices = POSEvaluator.evaluate(train_dataset.sentences(), [eager_tagger, viberti_tagger, mp_tagger])
    confusion_matrices = POSEvaluator.evaluate(test_dataset.sentences(), [eager_tagger, viberti_tagger, mp_tagger])


def run_all_evaluation():
    for dataset in prepare_datasets(TREE_BANKS_DIR):
        run_evaluation(dataset)


def main():
    pass


parser = ArgumentParser()
parser.add_argument("Run type", metavar="run_type", type=str, help="Run type: [test] or [eval]")
parser.add_argument("Dataset", metavar="dataset", type=str, help="Run type: [test] or [eval]")

args = parser.parse_args()
def run_program()


if __name__ == "__main__":
    main()
