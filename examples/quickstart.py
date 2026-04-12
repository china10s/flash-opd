"""FlashOPD Quickstart — 20 行代码启动 On-Policy Distillation."""
from flashopd import OPDConfig, OPDTrainer, create_teacher
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset

cfg = OPDConfig(
    student_model="Qwen/Qwen2.5-1.5B-Instruct",
    teacher_model="Qwen/Qwen2.5-7B-Instruct",
    kl_type="reverse",
    kl_coef=0.1,
    max_new_tokens=256,
)

tokenizer = AutoTokenizer.from_pretrained(cfg.student_model, trust_remote_code=True)
student = AutoModelForCausalLM.from_pretrained(cfg.student_model, torch_dtype="auto", trust_remote_code=True)
teacher = create_teacher(cfg, student_tokenizer=tokenizer)
dataset = load_dataset("json", data_files="data.jsonl", split="train")

trainer = OPDTrainer(
    opd_config=cfg,
    teacher=teacher,
    model=student,
    args=TrainingArguments(output_dir="./output", num_train_epochs=1, bf16=True),
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
