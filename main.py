from src.maskSiamese import MaskSiamese 
from src.trainer import *

if __name__ == "__main__":
    
    # ==================================== INITIAL SETUP ====================================

    model_dir = './mask_siamese_train_output/'
    makedir(model_dir)
    save_code_state(model_dir + "code/")

    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))

    # ==================================== LOAD DATASET ====================================
    
    train_loader = train_loader_helper(use_shuffle=True)
    test_loader = test_loader_helper(use_shuffle=False)

    log(f'training set size: {len(train_loader.dataset)}')
    log(f'test set size: {len(test_loader.dataset)}')
    log(f'batch size: {my_batch_size}')

    # ==================================== CONSTRUCT MODEL ====================================
            
    model = MaskSiamese()
    model = model.to(MY_GPU_DEVICE_NAME)
    multi_model = torch.nn.DataParallel(model)
    
    print("Loading Model with the following options:")
    print(f"Variant: {'Vector 49' if OPT_VECTOR49 else 'Matrix 49x49'}")
    print(f"Masking: {'Enabled' if OPT_USING_MASK else 'Disabled'}")
    if OPT_USING_MASK:
        print(f"Mask Placeholder: {OPT_PLACEHOLDER} with value {OPT_PLACEHOLDER_VAL if OPT_PLACEHOLDER in ['LEARNABLE', 'CONSTANT'] else 'N/A'}")

    # ==================================== SETUP OPTIMIZER ====================================

    my_optimizer = torch.optim.Adam([
        {'params': model.features.parameters(), 'lr': 1e-4, 'weight_decay': 1e-5},
        {'params': model.add_on_layers.parameters(), 'lr': 1e-3, 'weight_decay': 1e-5},
        {'params': model.last_layer.parameters(), 'lr': 1e-3},
        # {'params': model.mask_placeholder, 'lr': 1e-2, 'weight_decay': 0.0} # NEEDS TO BE UNCOMMENTED IF USING A LEARNABLE MASK PLACEHOLDER
    ])

    my_lr_scheduler = torch.optim.lr_scheduler.StepLR(my_optimizer, step_size = 5, gamma = 0.5)

    # ==================================== TRAIN MODEL ====================================
    
    trainer = Trainer(multi_model, log, (train_loader, test_loader), my_optimizer, my_lr_scheduler, model_dir)

    for epoch in range(my_num_train_epochs):
        log(f'epoch: \t{epoch}')

        trainer.train() # Train the model

        # Test the model and save it if it performs better
        acc, early_stop = trainer.test_save(epoch)
        
        # Check if the model's performance is not improving and stop training
        if early_stop:
            break
    
    logclose()