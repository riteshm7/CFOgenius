# Copyright 2023 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import datasets

_DESCRIPTION = """\
The dataset contains the annual report of US public firms filing with the SEC EDGAR system.
Each annual report (10K filing) is broken into 20 sections. Each section is split into individual sentences.
Sentiment labels are provided on a per filing basis from the market reaction around the filing data.
Additional metadata for each filing is included in the dataset.
"""

_LICENSE = "apache-2.0"

_NOS_SHARDS = 10

_URLS = {item+'_'+config: ["data/"+config+"/"+item+"/shard_"+str(shard)+".jsonl" for shard in range(_NOS_SHARDS)] for item in ['test', 'train', 'validate'] for config in ["large", "small"]}

_REPORT_KEYS = ['section_1', 'section_1A', 'section_1B', 
                 'section_2', 'section_3', 'section_4', 
                 'section_5', 'section_6', 'section_7', 
                 'section_7A', 'section_8', 'section_9', 
                 'section_9A', 'section_9B', 'section_10', 
                 'section_11', 'section_12', 'section_13', 
                 'section_14', 'section_15']

_LITE_FEATURES = ["cik", "sentence", "section", "labels", "filingDate", "docID", "sentenceID", "sentenceCount"]

_ALL_FEATURES = {
    "cik": datasets.Value("string"),
    "sentence": datasets.Value("string"),
    "section": datasets.ClassLabel(num_classes=20, 
                                   names=_REPORT_KEYS),
    "labels": {
            "1d": datasets.ClassLabel(num_classes=2, names=["positive", "negative"]),
            "5d": datasets.ClassLabel(num_classes=2, names=["positive", "negative"]),
            "30d": datasets.ClassLabel(num_classes=2, names=["positive", "negative"]),
        },
    "filingDate": datasets.Value("string"),
    "name": datasets.Value("string"), 
    "docID": datasets.Value("string"), 
    "sentenceID": datasets.Value("string"), 
    "sentenceCount": datasets.Value("int64"), 
    "tickers": [datasets.Value("string")],  
    "exchanges": [datasets.Value("string")],
    "entityType": datasets.Value("string"),
    "sic": datasets.Value("string"),
    "stateOfIncorporation": datasets.Value("string"),
    "tickerCount": datasets.Value("int32"),
    "acceptanceDateTime": datasets.Value("string"),
    "form": datasets.Value("string"),
    "reportDate": datasets.Value("string"),  
    "returns": {
            "1d": {
                    "closePriceEndDate": datasets.Value("float32"),
                    "closePriceStartDate": datasets.Value("float32"),
                    "endDate": datasets.Value("string"),
                    "startDate": datasets.Value("string"),
                    "ret": datasets.Value("float32"),
                },
            "5d": {
                    "closePriceEndDate": datasets.Value("float32"),
                    "closePriceStartDate": datasets.Value("float32"),
                    "endDate": datasets.Value("string"),
                    "startDate": datasets.Value("string"),
                    "ret": datasets.Value("float32"),
                },
            "30d": {
                    "closePriceEndDate": datasets.Value("float32"),
                    "closePriceStartDate": datasets.Value("float32"),
                    "endDate": datasets.Value("string"),
                    "startDate": datasets.Value("string"),
                    "ret": datasets.Value("float32"),
                }
        },
}


class FinancialReportsSec(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.1.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="large_lite", version=VERSION, description="This returns the dataset with only the critical data needed for analysis."),
        datasets.BuilderConfig(name="large_full", version=VERSION, description="This returns the dataset with all metadata included."),
        datasets.BuilderConfig(name="small_lite", version=VERSION, description="This returns a smaller version of the dataset with only the critical data needed for analysis."),
        datasets.BuilderConfig(name="small_full", version=VERSION, description="This returns a smaller version of the dataset with all metadata included."),
    ]

    def _info(self):        
        
        lite_features = datasets.Features({k: v for k, v in _ALL_FEATURES.items() if k in _LITE_FEATURES})
        full_features = datasets.Features(_ALL_FEATURES)
        
        features = full_features if self.config.name.endswith('full') else lite_features
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,
            # License for the dataset if available
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        if self.config.name.split('_')[0] == 'large':
            urls = {k: v for k, v in _URLS.items() if k.endswith('large')}
        else:
            urls = {k: v for k, v in _URLS.items() if k.endswith('small')}
            
        data_dir = dl_manager.download_and_extract(urls)
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepaths": data_dir["train_large"] if self.config.name.startswith('large') else data_dir["train_small"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepaths": data_dir["validate_large"] if self.config.name.startswith('large') else data_dir["validate_small"],
                    "split": "validate",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepaths": data_dir["test_large"] if self.config.name.startswith('large') else data_dir["test_small"],
                    "split": "test"
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepaths, split):
        # Store the sentence count for the current config. Not unique across configs.
        sentenceCount = 0
        
        # filepath is expected to be already parsed for config and split.
        for filepath in filepaths:           
            with open(filepath, encoding="utf-8") as f:                
                for firmIdx, row in enumerate(f):
                    # Each row is a single firm observation.
                    data = json.loads(row)                    
                    # Iterate the filings.
                    for filing in data["filings"]:
                        # Iterate over _REPORT_KEYS to ensure that the order of sections
                        # is consistent.
                        for section_id in _REPORT_KEYS:
                            # For the small versions of the dataset, some of the
                            # sections will be cut-off and hence None.
                            if filing["report"][section_id] is None:
                                return None
                            # Finally, iterate the sentences in the section of the filing.
                            for idx, sentence in enumerate(filing["report"][section_id]):  
                                sentenceCount += 1
                                key = data["cik"]+'_'+filing["form"]+'_'+filing["reportDate"].split('-')[0]+'_'+section_id+'_'+str(idx)
                                
                                if self.config.name.endswith('lite'):                                    
                                    yield key, {
                                        "cik": data["cik"],
                                        "sentence": sentence,
                                        "section": section_id,
                                        "labels": filing["labels"],
                                        "filingDate": filing["filingDate"],
                                        "docID": data["cik"]+'_'+filing["form"]+'_'+filing["reportDate"].split('-')[0],
                                        "sentenceID": key,
                                        "sentenceCount": sentenceCount,
                                    }
                                else:
                                    yield key, {
                                        "cik": data["cik"],
                                        "sentence": sentence,
                                        "section": section_id,
                                        "labels": filing["labels"],
                                        "filingDate": filing["filingDate"],
                                        "docID": data["cik"]+'_'+filing["form"]+'_'+filing["reportDate"].split('-')[0],
                                        "sentenceID": key,
                                        "sentenceCount": sentenceCount,
                                        "name": data["name"],
                                        "tickers": data["tickers"],
                                        "exchanges": data["exchanges"],
                                        "entityType": data["entityType"],
                                        "sic": data["sic"],
                                        "stateOfIncorporation": data["stateOfIncorporation"],
                                        "tickerCount": data["tickerCount"],
                                        "acceptanceDateTime": filing["acceptanceDateTime"],
                                        "form": filing["form"],
                                        "reportDate": filing["reportDate"],
                                        "returns": filing["returns"],
                                    }