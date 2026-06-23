
# ============================================================
# Document Q&A with Windows file picker
# Supports PDF and TXT files
# ============================================================

import os
import sys
import warnings
import logging
import tkinter as tk
from tkinter import filedialog

import torch
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MAX_DOCUMENT_CHARACTERS = 30_000
MAX_HISTORY_MESSAGES = 12


# ------------------------------------------------------------
# Open a Windows file-picker window
# ------------------------------------------------------------

def choose_file():
    root = tk.Tk()
    root.withdraw()

    try:
        root.attributes("-topmost", True)
    except tk.TclError:
        pass

    file_path = filedialog.askopenfilename(
        title="Select a PDF or text document",
        initialdir=os.path.expanduser("~/Downloads"),
        filetypes=[
            ("Supported documents", "*.pdf *.txt"),
            ("PDF files", "*.pdf"),
            ("Text files", "*.txt"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()

    if not file_path:
        raise SystemExit("No file was selected.")

    return file_path


# ------------------------------------------------------------
# Read text from a PDF or TXT file
# ------------------------------------------------------------

def load_text(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    extension = os.path.splitext(path)[1].lower()

    if extension == ".pdf":
        try:
            reader = PdfReader(path)
        except Exception as error:
            raise RuntimeError(f"Could not open the PDF: {error}") from error

        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
                pages.append(page_text)
            except Exception as error:
                print(
                    f"Warning: Could not extract page {page_number}: {error}"
                )

        document_text = "\n".join(pages).strip()

        if not document_text:
            raise RuntimeError(
                "No readable text was found in this PDF. "
                "It may contain scanned images instead of selectable text."
            )

        return document_text

    if extension == ".txt":
        try:
            with open(path, "r", encoding="utf-8") as file:
                return file.read().strip()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1") as file:
                return file.read().strip()

    raise ValueError("Only PDF and TXT files are supported.")


# ------------------------------------------------------------
# Shorten very large documents to avoid exceeding model memory
# ------------------------------------------------------------

def limit_document_size(document_text):
    if len(document_text) <= MAX_DOCUMENT_CHARACTERS:
        return document_text

    print(
        f"\nThe document contains {len(document_text):,} characters."
    )
    print(
        f"Using the first {MAX_DOCUMENT_CHARACTERS:,} characters "
        "to prevent memory problems."
    )

    return document_text[:MAX_DOCUMENT_CHARACTERS]


# ------------------------------------------------------------
# Create the system prompt
# ------------------------------------------------------------

def build_system_prompt(document_text):
    return (
        "You are a precise assistant that answers questions about the "
        "provided document.\n\n"
        "Rules:\n"
        "1. Use only information found in the document.\n"
        "2. Do not invent facts.\n"
        "3. If the answer is not stated, say that it is not stated.\n"
        "4. Keep answers concise but complete.\n\n"
        "=== DOCUMENT ===\n"
        f"{document_text}\n"
        "=== END DOCUMENT ==="
    )


# ------------------------------------------------------------
# Generate an answer from the local language model
# ------------------------------------------------------------

def generate_answer(
    tokenizer,
    model,
    messages,
    device,
    max_new_tokens=400,
):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[1]
    generated_tokens = output[0][input_length:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return answer.strip()


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------

def main():
    print("=" * 60)
    print("Document Q&A")
    print("=" * 60)

    try:
        document_path = choose_file()
    except Exception as error:
        print(f"File picker error: {error}")
        sys.exit(1)

    print(f"\nSelected file:\n{document_path}")

    try:
        document_text = load_text(document_path)
    except Exception as error:
        print(f"\nCould not read the document:\n{error}")
        sys.exit(1)

    document_text = limit_document_size(document_text)

    word_count = len(document_text.split())

    print(f"\nExtracted approximately {word_count:,} words.")

    # Use a larger model when a GPU is available.
    if torch.cuda.is_available():
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        device = "cuda"
        dtype = torch.float16
    else:
        # Smaller model is safer for laptops without an NVIDIA GPU.
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        device = "cpu"
        dtype = torch.float32

    print("\nLoading language model:")
    print(model_name)
    print(f"Device: {device.upper()}")
    print(
        "The first run will download the model and may take several minutes."
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        model.to(device)
        model.eval()

    except Exception as error:
        print("\nThe language model could not be loaded.")
        print(error)
        print("\nCheck that:")
        print("1. Your internet connection is working.")
        print("2. The required packages are installed.")
        print("3. Your computer has enough free memory.")
        sys.exit(1)

    system_prompt = build_system_prompt(document_text)

    history = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    print("\n" + "=" * 60)
    print(f"Loaded: {os.path.basename(document_path)}")
    print("=" * 60)

    try:
        summary = generate_answer(
            tokenizer=tokenizer,
            model=model,
            messages=history
            + [
                {
                    "role": "user",
                    "content": (
                        "Give a concise 3-4 sentence overview "
                        "of this document."
                    ),
                }
            ],
            device=device,
        )

        print("\nOVERVIEW:\n")
        print(summary)

    except Exception as error:
        print(f"\nCould not generate the overview: {error}")

    print("\nExample questions:")
    print("  • What are the main topics?")
    print("  • Summarize the key qualifications.")
    print("  • What skills are mentioned?")
    print("  • What is the most important information?")
    print()
    print("Commands:")
    print("  summary  - generate a new summary")
    print("  reset    - clear conversation memory")
    print("  quit     - close the program")
    print()

    # --------------------------------------------------------
    # Interactive chat loop
    # --------------------------------------------------------

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nDone.")
            break

        if not question:
            continue

        command = question.lower()

        if command in {"quit", "exit"}:
            print("Done.")
            break

        if command == "reset":
            history = [
                {
                    "role": "system",
                    "content": system_prompt,
                }
            ]

            print("\nConversation memory cleared.\n")
            continue

        if command == "summary":
            question = (
                "Give a fresh 3-4 sentence overview of the document."
            )

        history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        try:
            answer = generate_answer(
                tokenizer=tokenizer,
                model=model,
                messages=history,
                device=device,
            )

        except RuntimeError as error:
            print("\nThe model ran out of memory or encountered an error.")
            print(error)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            history.pop()
            continue

        history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        # Keep the system prompt plus only recent conversation messages.
        if len(history) > MAX_HISTORY_MESSAGES + 1:
            history = [
                history[0],
                *history[-MAX_HISTORY_MESSAGES:],
            ]

        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()

