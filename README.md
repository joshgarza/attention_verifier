# Attention-Guided Response Verification

This project, developed during the Llamacon Hackathon, explores using attention mechanisms in Large Language Models (LLMs) to verify the factual grounding of their responses, particularly in long-context question-answering scenarios. The core idea is to identify the specific text segments in a source document that the model "attended to" most when generating an answer, and then use this "evidence" to assess the answer's validity.

## Problem Addressed

LLMs can sometimes "hallucinate" or generate plausible-sounding but factually incorrect information, especially when dealing with large amounts of input text. This tool aims to provide a method for:

1.  Identifying the textual basis (evidence) for a model's generated answer.
2.  Verifying the answer against this identified evidence, both programmatically (NLI) and via human review.
3.  Flagging potential unsupported claims or hallucinations.

## Key Features

- **Evidence Identification:** Processes attention weights from Llama 3.3 70B (during answer generation) to pinpoint sentences in the source document most influential to the output.
- **NLI-based Verification:** Uses an LLM (Llama 4 API - Maverick 8B) to perform a Natural Language Inference (NLI) check (Entailment, Contradiction, Neutral) between the identified evidence and the generated answer.
- **Confidence Score:** Derives a numerical confidence score from the NLI judgment.
- **Interactive HTML Report:** Generates a static HTML report displaying the question, model's answer, NLI judgment, NLI reasoning, confidence score, and the full source document with top evidence sentences highlighted.
- **Flask Web Interface:** A simple two-step web UI to input a document/question, run inference, and then trigger the verification and report generation.

## How It Works (High-Level Pipeline)

1.  **Input:** User provides a source document and a question via the web interface.
2.  **Answer Generation & Attention Extraction:**
    - The local Llama 3.3 70B-Instruct (INT4 quantized) model generates an answer to the question based on the document.
    - Attention weights (currently focusing on last-token attention from `model.generate()`) are captured.
3.  **Attention Processing & Evidence Mapping:**
    - The captured attention scores are processed.
    - Attention is aggregated from tokens to sentences within the original document.
    - Sentences are ranked by their aggregated attention scores to identify top evidence.
4.  **NLI Verification:**
    - The top evidence sentences and the model's generated answer are formatted as a premise-hypothesis pair.
    - This pair is sent to the Llama 4 API (Maverick 8B) for an NLI judgment (Entailment, Contradiction, Neutral) and reasoning.
5.  **Report Generation:**
    - A confidence score is derived from the NLI judgment.
    - A static HTML report is generated, visually highlighting the evidence sentences within the context of the full document and displaying the NLI results.

## Technologies Used

- **Models:**
  - `meta-llama/Llama-3.3-70B-Instruct` (local, for answer generation & attention)
  - Llama 4 API (Maverick 8B, for NLI verification)
- **Core Libraries:**
  - Python 3.x
  - PyTorch
  - Hugging Face `transformers` (for model loading and generation)
  - `bitsandbytes` (for INT4 quantization)
  - `nltk` (for sentence tokenization)
- **Web & Reporting:**
  - Flask (for the web application)
  - Jinja2 (for HTML templating)
  - HTML, CSS, JavaScript (for the report and frontend)
- **APIs:**
  - `requests` (for Llama 4 API interaction)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/joshgarza/attention_verifier.git
    cd attention_verifier
    ```
2.  **Install dependencies:**
    (It's recommended to use a virtual environment)
    ```bash
    pip install -r requirements.txt
    # Note: Ensure PyTorch is installed compatible with your CUDA version if using GPU.
    # nltk data might be needed:
    # python -c "import nltk; nltk.download('punkt')"
    ```
3.  **Environment Variables:**
    - Set your Llama API key for the NLI check:
      ```bash
      export LLAMA_API_KEY='YOUR_LLAMA_API_KEY_HERE'
      ```
4.  **Model Access:**
    - Ensure you have access to `meta-llama/Llama-3.3-70B-Instruct` via Hugging Face Hub and are authenticated (`huggingface-cli login`). The model will be downloaded on first run if not cached.

## Usage

1.  **Run the Flask application:**
    ```bash
    python app.py
    ```
    The Llama 3 model will be loaded into memory (this may take some time on first startup).
2.  **Open your browser** and navigate to `http://127.0.0.1:5000/` (or the port specified in `app.py`).
3.  **Step 1: Run Inference:**
    - Paste your source document into the "Document" textarea.
    - Type your question into the "Question" input field.
    - Click "Run Inference & Get Answer". The model's answer will appear.
4.  **Step 2: Grade Response & Generate Report:**
    - Once the answer is displayed, click "Grade Response & Generate Report".
    - This will trigger attention processing, the NLI API call, and report generation.
    - A link to the `verification_report_[job_id].html` will appear. Click it to view the report.

## Example Output

The generated HTML report (`verification_report.html`) includes:

- The original Question.
- The Model's Answer.
- NLI Judgment (e.g., ENTAILMENT, CONTRADICTION, NEUTRAL).
- NLI Reasoning (from the Llama 4 API, in a collapsible section).
- Calculated Confidence Score.
- The full original Document text, with the top N evidence sentences highlighted (e.g., with a red gradient indicating attention intensity).
- A ranked list of the top N evidence sentences with their raw attention scores.

## Known Limitations

- **Llama 3 for Attention:** Uses Llama 3.3 for attention extraction due to initial difficulties loading Llama 4 models with attention output capabilities on the available hardware during the hackathon.
- **Last-Token Attention:** The current attention processing primarily relies on the attention scores from the last generated token(s) as output by `transformers` `model.generate(..., output_attentions=True)`. This is a simplification and doesn't capture the full attention dynamics across the entire generation sequence.
- **Approximations:** Tokenizer offset mapping and sentence boundary detection involve some inherent approximations.
- **NLI Model Dependency:** The NLI verification relies on the performance and behavior of the Llama 4 API model.
- **Scalability of Human Review:** While the tool aids human review, the review itself is manual.

## Future Work & Potential Captum Integration

This project demonstrates a practical approach to leveraging attention for response verification. Future work could explore:

- **Full Sequence Attention Analysis:** Moving beyond last-token attention to analyze attention patterns across all generated tokens.
- **Cross-Layer Attention Aggregation:** More sophisticated methods for combining attention information from different layers and heads.
- **Feature-Level Interpretability:** Inspired by work like Anthropic's on sparse autoencoders, investigating if specific "features" (interpretable directions in activation space) correlate with evidence selection or hallucination.
- **Guided Fine-Tuning:** Using the insights from attention and NLI verification to create datasets or signals for fine-tuning models to be more factually grounded and less prone to hallucination.

**Relevance to Captum:**
This project's focus on extracting and utilizing attention patterns for model understanding and verification aligns directly with Captum's mission to provide tools for interpreting PyTorch models. Potential contributions or extensions relevant to Captum could include:

- **Utilities for Generative Model Attention:** Developing and integrating Captum attributes or utilities specifically designed to handle attention outputs from `model.generate()` in `transformers`, which often have a different structure (e.g., per-generated-token) than typical classifier attention.
- **Source Text Mapping Tools:** Functions to robustly map attention scores from token-space back to meaningful units in the source text (words, sentences, spans), especially for long contexts.
- **Visualization for Generative Attention:** Adapting or creating visualizations suitable for showing how attention flows from generated tokens back to large source documents.
- **Benchmarking Attention-based Verification:** Providing a framework or examples within Captum to benchmark different attention processing techniques for tasks like evidence grounding.

This work could help Captum expand its support for interpreting the increasingly complex behaviors of large generative models.

## License

MIT License

---

_Developed by: Josh Garza_

_Hackathon: Llamacon 2025_
