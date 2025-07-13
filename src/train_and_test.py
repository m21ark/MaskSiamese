import time
import torch
import torch.nn.functional as F

from src.helpers import *

# ==================================== AUX FUNCS ====================================

def train(model, dataloader, optimizer, log=print):
    assert(optimizer is not None)
    log('\ttrain')
    model.train()
    return _train_or_test(model, dataloader, optimizer, log)

def test(model, dataloader, log=print):
    log('\ttest')
    model.eval()
    return _train_or_test(model, dataloader, None, log)

def _unwrap_model(model):
    while type(model) == torch.nn.DataParallel:
        model = model.module
    model = model.to(MY_GPU_DEVICE_NAME)
    return torch.nn.DataParallel(model)

# ==================================== MAIN FUNCS ====================================

def _train_or_test(model, dataloader, optimizer, log):

    is_train = optimizer is not None 
    multi_model = _unwrap_model(model)
    eps = multi_model.module.epsilon
    start_time = time.time()
    stats_num_batches = 0

    # Loss metrics
    stats_loss = 0
    cross_ent_func = torch.nn.BCEWithLogitsLoss()
    tp, tn, fp, fn = 0, 0, 0, 0

    for _, ((anchor_img, other_img), (anchor_mask, other_mask), label) in enumerate(dataloader):
        
        anchor_img = anchor_img.to(MY_GPU_DEVICE_NAME) # [b_size, 3, 224, 224]
        other_img = other_img.to(MY_GPU_DEVICE_NAME) # [b_size, 3, 224, 224]

        anchor_mask = anchor_mask.to(MY_GPU_DEVICE_NAME) # [b_size, 1, 7, 7]
        other_mask = other_mask.to(MY_GPU_DEVICE_NAME) # [b_size, 1, 7, 7]
        
        label = label.to(MY_GPU_DEVICE_NAME) # [b_size]

        with torch.set_grad_enabled(is_train):
            
            # ======================================== MODEL EXECUTE ========================================
            
            logits, distances = multi_model.forward(anchor_img, other_img, anchor_mask, other_mask) 
            
            # compute loss (convert binary labels to float)
            cross_entropy_cost = cross_ent_func(logits, label.float())
            batch_loss = cross_entropy_cost
            
            # ======================================== EVAlUATION STATISTICS ========================================

            predicted = torch.sigmoid(logits).round() # [b_size] --> Either 0.0 or 1.0 (0.5 threshold)
            
            tp += ((predicted == 1) & (label == 1)).sum().item()
            tn += ((predicted == 0) & (label == 0)).sum().item()
            fp += ((predicted == 1) & (label == 0)).sum().item()
            fn += ((predicted == 0) & (label == 1)).sum().item()
            
            stats_num_batches += 1
            
            # Loss metrics
            stats_loss += batch_loss.item()

        # ======================================== TRAINING OPTIMIZATION ========================================

        # backpropagate the loss and update the model weights
        if is_train:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
        # Only for local debugging
        if MY_GPU_DEVICE_NAME == 'mps':
            print(f'b {stats_num_batches}/{len(dataloader)} | l {batch_loss.item():.2f}', end='\r')
                
    # ======================================== LOG RESULTS ========================================
    
    log(f'\n\ttime: \t{(time.time() - start_time):.3f}')
    if OPT_USING_MASK and OPT_PLACEHOLDER == "LEARNABLE":
        log(f'\tmask_val: \t{multi_model.module.mask_placeholder.item():.3f}')
    
    # Performance Metrics 
    acc = (tp + tn) / (tp + tn + fp + fn)
    log(f'\tacc: \t{(acc * 100):.3f}%')
    log(f'\tprec: \t{(tp / (tp + fp + eps) * 100):.3f}%')
    log(f'\trec: \t{(tp / (tp + fn + eps) * 100):.3f}%')
    log(f'\tf1: \t{(2 * tp) / (2 * tp + fp + fn + eps) * 100:.3f}%')
    
    # Confusion matrix
    log(f'\ttn: \t{tn}')
    log(f'\tfp: \t{fp}')
    log(f'\tfn: \t{fn}')
    log(f'\ttp: \t{tp}')
    
    # Epoch Loss
    avg_print = lambda name, metric: log(f'\t{name}: \t{(metric / stats_num_batches):.3f}')
    avg_print('total_loss', stats_loss) # Only cross entropy loss
    log('-' * 50)
    
    return acc
