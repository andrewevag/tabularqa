# tabularqa

This repo includes the system developed for SemEval 2025 Task 8: Question Answering Over Tabular Data by AILS-NTUA.

The system ranked 1st in the proprietary models ranking in both subtasks of the competition.

The system description paper will be published in ACL 2025. The preprint is available at [arXiv](https://arxiv.org/abs/2503.00435).

## Architecture
![Architecture](./assets/Architecture.png)
The system performs Text-to-Python Code conversion of user queries through prompting Large Language Models (LLMs). More details on the architecture can be found in the paper.

## Usage Instructions

> [!NOTE] 
> Tested on Python 3.12

1. Clone the repository
2. Install the required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```
3. Set up credentials or models based on the evaluation scenario:

   - **For evaluating Claude 3.5 Sonnet or Llama 3.1 Instruct-405B:**
     Create a `.env` file in the root directory and add AWS credentials:
     ```bash
     AWS_ACCESS_KEY_ID=your_access_key_id
     AWS_SECRET_ACCESS_KEY=your_secret_access_key
     ```

   - **For evaluating Ollama models (`llama3.1:8b`, `llama3.3:70b`, `qwen2.5-coder:7b`):**
     Download the models by following the instructions on the [Ollama website](https://ollama.com/models). Ensure that Ollama is installed and running on port 11434.


4. Download `competition.zip` and extract it in the root directory for running the model in the [DataBench](https://huggingface.co/datasets/cardiffnlp/databench) Test Set (this is the default behavior, can be changed by loading another split as shown in the Hugging Face Page). This can be downloaded from the [DataBench Competition Page](https://www.codabench.org/competitions/3360/) or directly from [here](https://drive.google.com/file/d/1IpSi0gNPYj9a9lNbWPsL3TxIBILoLsfE/view?usp=sharing).
```bash
unzip competition.zip
```
5. Download the `answers.zip` file with the answers for the test set and extract it in `competition/answers/` directory.
```bash
wget https://raw.githubusercontent.com/jorses/databench_eval/main/examples/answers.zip
mkdir -p competition/answers
unzip answers.zip -d competition/answers
```
6. Run the `main.py` script with input the specification of the pipeline. All pipelines are found in the `config/` folder. Include the `--lite` flag to run on DataBench lite.
```bash
python main.py --pipeline config/claude3.5-sonnet 
# or
python main.py --pipeline config/claude3.5-sonnet --lite
```
7. The results will be saved in a new `results/` directory.
