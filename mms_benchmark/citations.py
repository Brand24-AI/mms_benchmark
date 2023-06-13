from typing import Dict, Union

import bibtexparser
from datasets import DatasetDict


def get_citations(dataset: DatasetDict, citation_as_dict: bool = True) -> Dict[str, Union[Dict, str]]:
    lines = dataset.citation.split("% Datasets: ")[1:]
    original_dataset_to_bibtex = {}
    for line in lines:
        original_datasets, citation = line.split("\n", maxsplit=1)
        dataset_list = original_datasets.split(", ")

        for dataset in dataset_list:
            original_dataset_to_bibtex[dataset] = bibtexparser.loads(citation).entries[0] if citation_as_dict else citation
    return original_dataset_to_bibtex
