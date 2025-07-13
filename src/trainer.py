import src.train_and_test as tnt
from src.helpers import *

class Trainer:       
 
    def __init__(self, multi_model, log, loaders, my_optimizer, my_lr_scheduler, model_dir):
        
        self.model = multi_model
        self.log = log
        
        train_loader, test_loader = loaders
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.my_optimizer = my_optimizer
        self.my_lr_scheduler = my_lr_scheduler

        self.model_dir = model_dir
        
        self.max_seen_acc = 0        
        self.early_stopper = EarlyStopping(my_early_stopping_patience, my_early_stopping_delta)

    def train(self):
        _ = tnt.train(model=self.model, dataloader=self.train_loader, optimizer=self.my_optimizer, log=self.log)
        self.my_lr_scheduler.step()
            
    # Test the model, see if performance improved and save the model if it did
    def _save_model(self, model_name, acc, unconditionalSave=False):
        if acc >= my_acc_threshold_to_save_model or unconditionalSave:
            self.log('\tSaving model with {0:.2f}%'.format(acc * 100))
            torch.save(self.model.module.state_dict(), f=os.path.join(self.model_dir, f'{model_name}{acc*100:.2f}.pth'))
        
    def test_save(self, epoch):
        
        # Test the model and if it performs better, save it
        acc = tnt.test(model=self.model, dataloader=self.test_loader, log=self.log)
        # if acc > self.max_seen_acc: # TEMPORARY: UNCONDITIONAL SAVE
        self._save_model(f'epoch_{epoch}_', acc) # No Push Epoch
        self.max_seen_acc = acc       
            
        # Stop training early if test accuracy isn't improving
        return acc, self.early_stopper(acc)

class EarlyStopping:
    # Defines an early stopping mechanism to stop training 
    # if the model isn't improvingby at least a certain  
    # threshold amount after a certain number of epochs
    
    def __init__(self, patience, min_delta, acc_mode=True):
        self.patience = patience
        self.min_delta = min_delta
        self.best_accu = 0 
        self.min_test_loss = float('inf')
        self.counter = 0
        self.acc_mode = acc_mode

    def __call__(self, curr_val):
        if self.acc_mode and (curr_val > self.best_accu + self.min_delta):
            self.best_accu = curr_val
            self.counter = 0  # Reset patience
        elif not self.acc_mode and (curr_val < self.min_test_loss - self.min_delta):
            self.min_test_loss = curr_val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered! Stopping training.")
                return True
        return False
