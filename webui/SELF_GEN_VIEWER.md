# Self-Gen Data Viewer

Thanks Claude.

Running the viewer
```bash
uv run self_gen_viewer.py
```

Then open your browser and go to: **http://localhost:5001**

## Usage

1. **Select a Model Folder**: Choose from the dropdown list (e.g., `google/gemma-2-2b-it_temp_0.0_closed_qa_prob_1.0`)
2. **Select a Parquet File**: Once a folder is selected, available parquet files will appear
3. **Set Number of Samples**: Adjust the sample count (default: 100, max: 1000)
4. **Click "Load Data"**: View the visualized data with context and Q&A pairs

## Data Structure

The viewer expects data in the following structure:
```
data/raw_datasets/self_gen/
├── google/
│   └── gemma-2-2b-it_temp_0.0_closed_qa_prob_1.0/
│       └── fw_qa_v2/
│           └── *.parquet
└── mistralai/
    └── Mistral-7B-Instruct-v0.2_temp_0.0_closed_qa_prob_1.0/
        └── *.parquet
```
