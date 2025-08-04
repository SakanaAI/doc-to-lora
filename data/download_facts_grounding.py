# Install dependencies as needed:
# pip install kagglehub[hf-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "data/raw_datasets/FACTS/ds.parquet"

# Load the latest version
hf_dataset = kagglehub.load_dataset(
    KaggleDatasetAdapter.HUGGING_FACE,
    "deepmind/FACTS-grounding-examples",
    file_path,
    # Provide any additional arguments like
    # sql_query, hf_kwargs, or pandas_kwargs. See
    # the documenation for more information:
    # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterhugging_face
)

print("Hugging Face Dataset:", hf_dataset)
