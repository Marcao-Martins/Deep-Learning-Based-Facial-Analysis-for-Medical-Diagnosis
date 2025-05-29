import os
from typing import Any, Dict, List


class DataLoader:
    """
    Base DataLoader class to handle dataset loading.
    """
    def __init__(self, dataset_path: str):
        """
        Initialize with path to the dataset directory.
        """
        self.dataset_path = dataset_path

    def load_data(self) -> Any:
        """
        Load and return data required for the pipeline.
        Must be implemented in a subclass.
        """
        raise NotImplementedError("DataLoader.load_data must be implemented in a subclass.")

# Subclass for the AutismDataset
class AutismDataLoader(DataLoader):
    """
    DataLoader for the AutismDataset. Loads train, valid, and test splits
    and returns file paths and corresponding labels.
    """
    def __init__(self, dataset_path: str,
                 splits: List[str] = ["train", "valid", "test"],
                 classes: List[str] = ["Autistic", "Non_Autistic"]):
        super().__init__(dataset_path)
        self.splits = splits
        self.classes = classes

    def load_data(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Load image file paths and labels for each split.
        Returns a dict:
        {
            split_name: {
                "file_paths": List[str],
                "labels": List[int]
            }
        }
        """
        data: Dict[str, Dict[str, List[str]]] = {}
        for split in self.splits:
            split_path = os.path.join(self.dataset_path, split)
            if not os.path.isdir(split_path):
                continue
            file_paths: List[str] = []
            labels: List[int] = []
            for idx, cls in enumerate(self.classes):
                cls_dir = os.path.join(split_path, cls)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        file_paths.append(os.path.join(cls_dir, fname))
                        labels.append(idx)
            data[split] = {"file_paths": file_paths, "labels": labels}
        return data 