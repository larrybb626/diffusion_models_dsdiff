import lightning.pytorch as pl

class CheckPointSavingMetric(pl.Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint) :
        checkpoint["best_metric"]=pl_module.best_val_mae
        checkpoint["best_val_epoch"] = pl_module.best_val_epoch
        checkpoint["lr_list"] = pl_module.lr_list
        checkpoint["metric_values"] = pl_module.metric_values
        checkpoint["traning_loss"]=pl_module.traning_loss

class PrintInLog(pl.Callback):
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        print("call_back")
