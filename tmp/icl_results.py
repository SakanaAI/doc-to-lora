import json
from pathlib import Path

all_results_path = Path(
    "eval_results/google/gemma-2-2b-it/20250918-111058_734998c4/all_results.json"
)
all_result_modulated_path = Path(
    "../train_outputs/runs/Aug10_08-29-46_sakura-003_2000_f73050aa/eval-results-100000/all_results.json"
)
with open(all_results_path) as f:
    all_results = json.load(f)

with open(all_result_modulated_path) as f:
    all_results_modulated = json.load(f)

# Find keys with test_metaicl_macro_f1 and test_metaicl_no_context_macro_f1 prefixes
metaicl_keys = [k for k in all_results.keys() if k.startswith("test_metaicl_macro_f1")]
no_context_keys = [
    k for k in all_results.keys() if k.startswith("test_metaicl_no_context_macro_f1")
]

# Sort keys for consistent ordering
metaicl_keys.sort()
no_context_keys.sort()

print("Comparison of MetaICL macro F1 scores:")
print("=" * 140)
print(f"{'Without Context':<40} {'With Context':<40} {'Modulated (No Context)':<40}")
print("-" * 140)

# Create a mapping of task names to values for easier comparison
metaicl_dict = {}
no_context_dict = {}
modulated_dict = {}

for key in metaicl_keys:
    task_name = key.replace("test_metaicl_macro_f1_", "")
    metaicl_dict[task_name] = all_results[key]

for key in no_context_keys:
    task_name = key.replace("test_metaicl_no_context_macro_f1_", "")
    no_context_dict[task_name] = all_results[key]

# Get modulated results (assuming they use the same key pattern as with context)
for key in all_results_modulated.keys():
    if key.startswith("test_metaicl_macro_f1_"):
        task_name = key.replace("test_metaicl_macro_f1_", "")
        modulated_dict[task_name] = all_results_modulated[key]

# Get all unique task names from the base results (this determines ordering)
all_tasks = set(metaicl_dict.keys()) | set(no_context_dict.keys())

# Create list of tasks with their differences for sorting (based on without context vs with context)
task_data = []
for task in all_tasks:
    with_context = metaicl_dict.get(task, "N/A")
    without_context = no_context_dict.get(task, "N/A")
    modulated = modulated_dict.get(task, "N/A")

    if isinstance(with_context, (int, float)) and isinstance(
        without_context, (int, float)
    ):
        diff_with_context = with_context - without_context
        modulated_diff = (
            modulated - without_context
            if isinstance(modulated, (int, float))
            else float("-inf")
        )
        task_data.append(
            (
                task,
                without_context,
                with_context,
                modulated,
                diff_with_context,
                modulated_diff,
            )
        )
    else:
        # Put tasks with missing data at the end
        task_data.append(
            (
                task,
                without_context,
                with_context,
                modulated,
                float("-inf"),
                float("-inf"),
            )
        )

# Sort by difference (with context - without context) from high to low
task_data.sort(key=lambda x: x[4], reverse=True)

for (
    task,
    without_context,
    with_context,
    modulated,
    diff_with_context,
    modulated_diff,
) in task_data:
    if isinstance(with_context, (int, float)) and isinstance(
        without_context, (int, float)
    ):
        modulated_str = (
            f"{modulated:.4f}"
            if isinstance(modulated, (int, float))
            else str(modulated)
        )
        modulated_diff_str = (
            f"(+{modulated_diff:+.4f})"
            if isinstance(modulated_diff, (int, float))
            and modulated_diff != float("-inf")
            else ""
        )
        print(
            f"{task}: {without_context:.4f} {task}: {with_context:.4f} (+{diff_with_context:+.4f}) {task}: {modulated_str} {modulated_diff_str}"
        )
    else:
        print(f"{task}: {without_context} {task}: {with_context} {task}: {modulated}")

IMPROVEMENT_THRESHOLD = 0.10
# Filter tasks where with_context shows >5% increase over without_context
high_improvement_tasks = []
for (
    task,
    without_context,
    with_context,
    modulated,
    diff_with_context,
    modulated_diff,
) in task_data:
    if (
        isinstance(with_context, (int, float))
        and isinstance(without_context, (int, float))
        and isinstance(diff_with_context, (int, float))
        and diff_with_context > IMPROVEMENT_THRESHOLD
    ):
        high_improvement_tasks.append(
            (
                task,
                without_context,
                with_context,
                modulated,
                diff_with_context,
                modulated_diff,
            )
        )

if high_improvement_tasks:
    print("\n" + "=" * 80)
    print(
        f"Tasks with >{IMPROVEMENT_THRESHOLD * 100}% improvement when adding context ({len(high_improvement_tasks)} tasks):"
    )
    print("=" * 80)

    # Calculate averages
    with_context_improvements = [diff for _, _, _, _, diff, _ in high_improvement_tasks]
    modulated_improvements = [
        mod_diff
        for _, _, _, _, _, mod_diff in high_improvement_tasks
        if isinstance(mod_diff, (int, float)) and mod_diff != float("-inf")
    ]

    avg_with_context_improvement = sum(with_context_improvements) / len(
        with_context_improvements
    )
    avg_modulated_improvement = (
        sum(modulated_improvements) / len(modulated_improvements)
        if modulated_improvements
        else 0
    )

    print(f"Average improvement with context: +{avg_with_context_improvement:.4f}")
    print(f"Average improvement with modulated: +{avg_modulated_improvement:.4f}")

    print("\nDetailed breakdown:")
    for (
        task,
        without_context,
        with_context,
        modulated,
        diff_with_context,
        modulated_diff,
    ) in high_improvement_tasks:
        modulated_str = (
            f"{modulated:.4f}"
            if isinstance(modulated, (int, float))
            else str(modulated)
        )
        modulated_diff_str = (
            f"(+{modulated_diff:+.4f})"
            if isinstance(modulated_diff, (int, float))
            and modulated_diff != float("-inf")
            else ""
        )
        print(
            f"  {task}: {without_context:.4f} → {with_context:.4f} (+{diff_with_context:+.4f}) | Modulated: {modulated_str} {modulated_diff_str}"
        )
else:
    print(
        f"\nNo tasks found with >{IMPROVEMENT_THRESHOLD * 100}% improvement when adding context."
    )
