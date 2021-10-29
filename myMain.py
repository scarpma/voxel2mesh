print("Create network")
classifier = network(cfg)
classifier.cuda()

print("Initialize optimizer")
optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                              classifier.parameters()),
                       lr=cfg.learning_rate)

print("Load pre-processed data")
data_obj = cfg.data_obj
data = data_obj.quick_load_data(cfg, trial_id)

loader = DataLoader(data[DataModes.TRAINING],
                    batch_size=classifier.config.batch_size,
                    shuffle=True)

print("Trainset length: {}".format(loader.__len__()))

print("Initialize evaluator")
evaluator = Evaluator(classifier, optimizer, data, trial_path, cfg, data_obj)

print("Initialize trainer")
trainer = Trainer(classifier, loader, optimizer, cfg.numb_of_itrs,
                  cfg.eval_every, trial_path, evaluator)

trainer.train(start_iteration=epoch)
# To evaluate a pretrained model, uncomment line below and comment the line above
# evaluator.evaluate(epoch)
