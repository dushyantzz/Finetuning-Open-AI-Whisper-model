import os
import argparse
import logging
import numpy as np
from datasets import load_from_disk
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
import evaluate
from tqdm import tqdm
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

processor = None

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {
        "wer": wer,
        "cer": cer
    }

def prepare_dataset(examples):
    if "audio" in examples and isinstance(examples["audio"], dict) and "array" in examples["audio"]:
        audio = examples["audio"]
        examples["input_features"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
    elif "path" in examples:
        try:
            import torchaudio
            import torch

            waveform, sample_rate = torchaudio.load(examples["path"])

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

            examples["input_features"] = processor(
                waveform.squeeze().numpy(), sampling_rate=16000
            ).input_features[0]
        except Exception as e:
            import numpy as np
            logger.warning(f"Error loading audio from {examples['path']}: {e}")
            logger.warning("Using dummy input features")
            examples["input_features"] = np.zeros((80, 3000))
    else:
        import numpy as np
        logger.warning("No audio data found in examples. Using dummy input features")
        examples["input_features"] = np.zeros((80, 3000))

    examples["labels"] = processor(text=examples["sentence"]).input_ids
    return examples

def train_model(dataset_dict, model_name="openai/whisper-small", output_dir="./models/whisper-fine-tuned",
               learning_rate=1e-5, batch_size=8, num_epochs=3, dialect=None, resume_from_checkpoint=None):
    global processor

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading model: {model_name}")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    logger.info("Processing datasets")
    processed_datasets = {}

    for split, dataset in dataset_dict.items():
        logger.info(f"Processing {split} split with {len(dataset)} examples")
        processed_datasets[split] = dataset.map(
            prepare_dataset,
            remove_columns=dataset.column_names,
            desc=f"Processing {split} split",
            num_proc=4
        )

    train_dataset = processed_datasets["train"]
    num_train_samples = len(train_dataset)
    steps_per_epoch = num_train_samples // batch_size
    total_training_steps = steps_per_epoch * num_epochs

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        warmup_steps=steps_per_epoch,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="steps",
        eval_steps=steps_per_epoch // 2,
        logging_steps=steps_per_epoch // 10,
        save_steps=steps_per_epoch,
        save_total_limit=3,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    data_collator = DataCollatorForSeq2Seq(processor.tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=processed_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.tokenizer,
    )

    logger.info(f"Starting training with {num_train_samples} examples")
    logger.info(f"Training for {num_epochs} epochs with batch size {batch_size}")
    logger.info(f"Total training steps: {total_training_steps}")
    if dialect:
        logger.info(f"Training on dialect: {dialect}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    logger.info("Evaluating on test set")
    test_results = trainer.evaluate(processed_datasets["test"])
    logger.info(f"Test results: {test_results}")

    final_model_path = f"{output_dir}-final"
    logger.info(f"Saving final model to {final_model_path}")
    model.save_pretrained(final_model_path)
    processor.save_pretrained(final_model_path)

    with open(os.path.join(final_model_path, "test_results.txt"), "w") as f:
        for key, value in test_results.items():
            f.write(f"{key}: {value}\n")

    logger.info("Training completed and model saved!")

    return model, processor

def train_for_multiple_dialects(dataset_path, dialects, model_name="openai/whisper-small",
                               base_output_dir="./models/whisper-fine-tuned"):
    dialect_models = {}

    for dialect in dialects:
        logger.info(f"Training model for dialect: {dialect}")

        try:
            dataset_dict = {}
            for split in ["train", "validation", "test"]:
                dataset = load_from_disk(os.path.join(dataset_path, split))
                dataset = dataset.filter(lambda example: example["accent"] == dialect)
                dataset_dict[split] = dataset
                logger.info(f"Loaded {len(dataset)} examples for {dialect} dialect ({split} split)")
        except Exception as e:
            logger.error(f"Error loading dataset for dialect {dialect}: {e}")
            continue

        if len(dataset_dict["train"]) < 100:
            logger.warning(f"Not enough training data for dialect {dialect} (only {len(dataset_dict['train'])} examples). Skipping.")
            continue

        output_dir = f"{base_output_dir}-{dialect}"
        model, _ = train_model(
            dataset_dict=dataset_dict,
            model_name=model_name,
            output_dir=output_dir,
            dialect=dialect
        )

        dialect_models[dialect] = f"{output_dir}-final"

    return dialect_models

def main():
    parser = argparse.ArgumentParser(description="Train ASR model")
    parser.add_argument("--dataset_path", type=str, default="./data/processed_common_voice")
    parser.add_argument("--model_name", type=str, default="openai/whisper-small")
    parser.add_argument("--output_dir", type=str, default="./models/whisper-fine-tuned")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--dialect", type=str, default=None)
    parser.add_argument("--multi_dialect", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        dataset_dict = {}
        for split in ["train", "validation", "test"]:
            try:
                dataset_dict[split] = load_from_disk(os.path.join(args.dataset_path, split))
                logger.info(f"Loaded {len(dataset_dict[split])} examples from {os.path.join(args.dataset_path, split)}")
            except Exception as e:
                logger.error(f"Error loading {split} split: {e}")

                if split == "train":
                    logger.info("Preparing dataset from scratch")
                    from data_preparation import prepare_common_voice_dataset, prepare_dataset_for_training

                    logger.info("Creating example dataset for testing")
                    import numpy as np
                    import random
                    from datasets import Dataset

                    if args.dialect:
                        dialects = [args.dialect]
                    else:
                        dialects = ["us", "england", "indian", "australia"]

                    example_data = []
                    for i in range(100):
                        example_data.append({
                            "path": f"example_{i}.mp3",
                            "audio": {
                                "path": f"example_{i}.mp3",
                                "array": np.zeros(16000),
                                "sampling_rate": 16000
                            },
                            "sentence": f"This is example sentence {i}",
                            "accent": random.choice(dialects),
                        })

                    train_size = int(0.8 * len(example_data))
                    val_size = int(0.1 * len(example_data))

                    train_data = example_data[:train_size]
                    val_data = example_data[train_size:train_size+val_size]
                    test_data = example_data[train_size+val_size:]

                    processed_dataset = {
                        'train': Dataset.from_list(train_data),
                        'validation': Dataset.from_list(val_data),
                        'test': Dataset.from_list(test_data)
                    }

                    logger.info(f"Created example dataset with {len(train_data)} training, {len(val_data)} validation, and {len(test_data)} test examples")

                    os.makedirs(args.dataset_path, exist_ok=True)
                    for s, ds in processed_dataset.items():
                        ds.save_to_disk(os.path.join(args.dataset_path, s))

                    dataset_dict = {}
                    for s in ["train", "validation", "test"]:
                        dataset_dict[s] = load_from_disk(os.path.join(args.dataset_path, s))

                    break

        if args.dialect and not args.multi_dialect:
            for split in dataset_dict:
                dataset_dict[split] = dataset_dict[split].filter(
                    lambda example: example["accent"] == args.dialect
                )
                logger.info(f"Filtered {split} split to {len(dataset_dict[split])} examples with dialect '{args.dialect}'")

        if args.multi_dialect:
            dialects = set(dataset_dict["train"]["accent"])
            logger.info(f"Training separate models for dialects: {dialects}")

            dialect_models = train_for_multiple_dialects(
                dataset_path=args.dataset_path,
                dialects=dialects,
                model_name=args.model_name,
                base_output_dir=args.output_dir
            )

            logger.info(f"Trained models for dialects: {list(dialect_models.keys())}")

        else:
            train_model(
                dataset_dict=dataset_dict,
                model_name=args.model_name,
                output_dir=args.output_dir,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                dialect=args.dialect,
                resume_from_checkpoint=args.resume_from_checkpoint
            )

    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
