import os
from typing import Any, List
import pandas as pd
from sklearn.model_selection import train_test_split


class RawInputArtifact:
    OUTPUT_FLAGS = ['train', 'val', 'test']
    def __init__(self, input_path, artifact_dir, train_split, val_split, test_split, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.input_path = input_path
        self.artifact_dir = artifact_dir
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.output_artifacts = []
        self.seed = seed

    def load_input_artifacts(self) -> Any:
        dataframes = []
        for item in self.output_artifacts:
            dataframes.append(pd.read_csv(item['path'], index_col=0, parse_dates=True))
        return dataframes
    
    def get_train_val_test_datasets(self) -> List[pd.DataFrame]:
        raw_input_df = raw_input_df = pd.read_csv(self.input_path, index_col=0, parse_dates=True)

        train_df, val_and_test = train_test_split(raw_input_df, test_size=(self.val_split + self.test_split), shuffle=False, random_state=self.seed)
        val_df, test_df = train_test_split(val_and_test, test_size=self.test_split / (self.val_split + self.test_split), shuffle=False, random_state=self.seed)

        return [train_df, val_df, test_df]
    
    def write_split_to_artifact(self):
        datasets = self.get_train_val_test_datasets()

        if not os.path.exists(f'{self.artifact_dir}/example_gen'):
            os.makedirs(f'{self.artifact_dir}/example_gen')

        for i, (flag, dataset) in enumerate(zip(self.OUTPUT_FLAGS, datasets)):

            output_path = f'{self.artifact_dir}/example_gen/{flag}.csv'
            dataset.to_csv(output_path)
            self.output_artifacts.append({
                'name': f'{flag}__split',
                'path': output_path
            })

    def run(self, **kwargs):
        self.write_split_to_artifact()
        return self.output_artifacts