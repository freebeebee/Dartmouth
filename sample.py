# Checkpointing and Configuring the dataset
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    set_seed(training_args.seed)
    train_raw_datasets = datasets.load_from_disk(data_args.train_file)
    valid_raw_datasets = datasets.load_from_disk(data_args.validation_file)

    personality_dic = {
    #    Due to the line limit, the code is omitted
    }
        
    lm_datasets = raw_datasets.map(
        #Code related to raw data processing is omitted due to the line limit
    )
# Cache for later use
    if not os.path.exists(f'./cache/lm_cache_{block_size}.pkl'):
        with open(f'./cache/lm_cache_{block_size}.pkl', 'wb') as f:
            pickle.dump(lm_datasets, f)
            
# Config loading and Model instantiation
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.pad_token_id = tokenizer.pad_token_id
    config.num_labels = 16
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config)
    model.resize_token_embeddings(len(tokenizer))

    
    if training_args.do_train:
        train_dataset = lm_datasets["train"]

    if training_args.do_eval:
        eval_dataset = lm_datasets["validation"]
        metric = evaluate.load("accuracy")

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            print(preds, labels)
            return metric.compute(predictions=preds, references=labels)
        
    trainer = Trainer(
        # Due to the line limit, somes code is omitted
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics 
    )

    # start traning or evaluating
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    
    if training_args.do_eval:
        from sklearn.metrics import(
            confusion_matrix, 
            accuracy_score, 
            f1_score, precision_score, recall_score
        )
        # print out the final result
        def compute_metrics_last(eval_preds):
            preds, labels = eval_preds 
            print(f"Confusion Matrix = \n{confusion_matrix(preds, labels)}")
            return{
                'Accuracy' : accuracy_score(preds, labels),
                'Precision' : precision_score(preds, labels, average='macro'),
                'Recall' : recall_score(preds, labels, average='macro'),
                'F1' : f1_score(preds, labels, average='macro')
            }
        trainer.compute_metrics = compute_metrics_last
        
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    trainer.create_model_card(**kwargs)