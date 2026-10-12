# Direct-Revision-or-Targeted-Replay-Comparing-Factual-Error-Repair-Policies-for-Long-Form-Generation

This repository contains the experiment code, datasets, and analysis notebook for comparing factual-error repair in Direct and Decomposed LLM workflows.

## Workflows

We evaluate four configurations:

- **Direct without HDM:** A single agent generates the output.
- **Decomposed without HDM:** Specialized agents plan, execute subtasks, and synthesize the output.
- **Direct with HDM:** A Hallucination Detection Module (HDM) identifies factual errors and guides final-output revision.
- **Decomposed with HDM:** HDM uses execution provenance to guide subtask replay and synthesis repair.

## Tasks and Datasets

- **Public-Event Narrative Synthesis (PNS):** 55 events with source articles.
- **Project-Grounded Research Ideation (PRI):** 100 research project records.
- **Scientific Related-Work Generation (RWG):** 100 Multi-XScience instances.

The datasets are provided as CSV or JSONL files. Python scripts are named by workflow, HDM setting, task, and model.

## Models

The provided scripts are configured for Gemma 3. The same workflows can be used with other models by updating all model-name settings and ensuring that the selected model is available through Ollama. Compatibility with the scripts’ structured JSON output requirements should be checked before running experiments.

## Analysis

The analysis notebook examines:

- Factual-error repair effectiveness.
- Attributed error origins and intervention routes.
- Final output quality.
- Runtime and repair efficiency.

## Usage

1. Install the Python packages imported by the scripts and notebook.
2. Start Ollama and make the configured model available.
3. Update the dataset paths, model settings, and output directories in the selected script.
4. Run the script using Python.
5. Update the input result paths in the analysis notebook and run its cells in order.

The analysis notebook requires experiment result CSVs in addition to the input datasets.
