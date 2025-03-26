
import fiftyone as fo
import fiftyone.zoo as foz

name = "BDDDataset"
data_path = "C:\\Personal\\Bosch_assignment\\assignment_data_bdd_files\\bdd100k_images_100k\\bdd100k\\images\\100k\\val"
labels_path = "C:\\Personal\\Bosch_assignment\\assignment_data_bdd_files\\bdd100k_labels_release\\bdd100k\\labels\\bdd100k_labels_images_val.json"

dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.BDDDataset,
    data_path=data_path,
    labels_path=labels_path,
    name=name,)

# sample = dataset.first()
session = fo.launch_app(dataset)
