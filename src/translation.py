
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

MODEL_NAME = "facebook/nllb-200-distilled-600M"
ADAPTER_PATH = "models/swahili_nllb_lora_fixed"

# Select GPU if available, otherwise use CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_translation_model():
    """Load the NLLB model and our fine-tuned LoRA adapter."""

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenizer.src_lang = "swh_Latn"
    tokenizer.tgt_lang = "eng_Latn"

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )

    base_model = base_model.to(DEVICE)

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH
    )

    model.eval()

    return model, tokenizer


def translate_text(text, model, tokenizer):
    """Translate a Kiswahili sentence into English."""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_length=128,
            num_beams=4
        )

    translation = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return translation
