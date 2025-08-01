from transformers import TrainingArguments, Trainer
from config import settings
class BertTrainer:
    def __init__(self, model, tokenizer, compute_metrics=None):
        self.model=model
        self.tokenizer=tokenizer
        self.compute_metrics=compute_metrics
    def get_training_args(self):
        """
        output_dir: folder where the model will be saved
        num_train: How many times will the training be repeated (number of epochs)
        
        """
        return TrainingArguments(
            output_dir=settings.MODEL_DIR,
            num_train_epochs=settings.EPOCHS,
            per_device_eval_batch_size=settings.BATCH_SIZE,
            per_device_train_batch_size=settings.BATCH_SIZE,
            learning_rate=settings.LEARNING_RATE,
            logging_dir=str(settings.MODEL_DIR/"logs"),
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=2,
            report_to="none"
        )
    def train(self, train_dataset, eval_dataset=None):
        training_args=self.get_training_args()
        trainer=Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics
        )
        try:
            trainer.train()
            print("success!")
            return trainer
        except Exception as e:
            print(e)
            raise
    