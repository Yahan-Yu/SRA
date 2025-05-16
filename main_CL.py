import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random
import scipy
import math
import pdb
from copy import deepcopy

from src.utils import *
from src.dataloader import *
from src.trainer import *
from src.model import *
from src.config import *
from src.Noise_G import *

import time


def main_cl(params):
    # pdb.set_trace()
    # ===========================================================================
    # Using Fixed Random Seed
    # 固定随机种子
    if params.seed: 
        random.seed(params.seed)
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        torch.cuda.manual_seed(params.seed)
        torch.backends.cudnn.deterministic = True
        
    # Initialize Experiment
    logger = init_experiment(params, logger_filename=params.logger_filename) # save logs
    logger.info(params.__dict__) # print the current config

    # Set domain name
    domain_name = os.path.basename(params.data_path[0]) # dataset name
    if domain_name=='':
        # Remove the final char '\' in the path
        domain_name = os.path.basename(params.data_path[0][:-1])

    # Generate Dataloader 
    ner_dataloader = NER_dataloader(data_path=params.data_path,
                                    domain_name=domain_name,
                                    batch_size=params.batch_size, 
                                    entity_list=params.entity_list,
                                    n_samples=params.n_samples,
                                    is_filter_O=params.is_filter_O,
                                    schema=params.schema,
                                    is_load_disjoin_train=params.is_load_disjoin_train)

    label_list = ner_dataloader.label_list # label list
    entity_list = ner_dataloader.entity_list # entity list
    num_classes_all = len(ner_dataloader.entity_list) # entity/classification amount
    pad_token_id = ner_dataloader.auto_tokenizer.pad_token_id # padding token's id = 0
    class_per_entity = len(params.schema)-1 # 2  ebery entity's label amount B- I-
    
    # Initialize the model for the first group of classes
    if params.model_name in ['bert-base-cased','roberta-base','bert-base-chinese']:
        # BERT-based NER Tagger
        model = BertTagger(output_dim=(1+class_per_entity*params.nb_class_fg), params=params)
    else:
        raise Exception('model name %s is invalid'%params.model_name)

    model.cuda()
    trainer = BaseTrainer(params, model, label_list)
    trainer.pad_token_id = pad_token_id # padding token's id = 0

    # ===========================================================================
    # Start training
    total_iter = int((num_classes_all-params.nb_class_fg)/params.nb_class_pg)+1 # number of tasks or steps
    assert (num_classes_all-params.nb_class_fg)%params.nb_class_pg==0, "Invalid class number!"

    trainer.inital_nb_classes = 1+class_per_entity*params.nb_class_fg
    trainer.nb_classes = num_classes_all*class_per_entity + 1
    trainer.classes = []

    for iteration in range(total_iter): # Traverse each step

        logger.info("=========================================================")   
        logger.info("Begin training the %d-th iter (total %d iters)"%(iteration+1, 
                                                                        total_iter))     
        logger.info("=========================================================")
        
        best_model_ckpt_name = "best_finetune_domain_%s_iteration_%d.pth"%(
                                domain_name, 
                                iteration) # current dataset, each task, name of the best model (based on current step/task's val dataset to choose)
        best_model_ckpt_path = os.path.join(
            params.dump_path, 
            best_model_ckpt_name
        ) # ./experiments/exp_name/exp_id/best_model_ckpt_name

        best_e_model_ckpt_name = "best_noise_finetune_domain_%s_iteration_%d.pth"%(
                                domain_name, 
                                iteration) # current dataset, each task, name of the best model (based on current step/task's val dataset to choose)
        best_e_model_ckpt_path = os.path.join(
            params.dump_path, 
            best_e_model_ckpt_name
        ) # ./experiments/exp_name/exp_id/best_model_ckpt_name

        if params.is_load_common_first_model: # True
            # under the same setting, base model only needs training once, the rest directly loads
            common_first_model_ckpt_name = "best_finetune_domain_%s_iteration_%d_fg_%d.pth"%(
                                    domain_name, 
                                    iteration,
                                    params.nb_class_fg)
            common_first_model_ckpth_path = os.path.join(
                os.path.dirname(os.path.dirname(params.dump_path)),
                common_first_model_ckpt_name
            ) 

        # Initialize a new model
        if params.is_from_scratch or iteration == 0:
            # Initialize the model for the first group of classes
            if params.model_name in ['bert-base-cased','roberta-base','bert-base-chinese']:
                # BERT-based NER Tagger the output layer changes dynamicly
                model = BertTagger(output_dim=(1+class_per_entity*(params.nb_class_fg+iteration*params.nb_class_pg)), params=params)
            else:
                raise Exception('model name %s is invalid'%params.model_name)
            
            trainer.model = model
            trainer.model.cuda()

            trainer.refer_model = None
            hidden_dim = trainer.model.classifier.hidden_dim # 768
            output_dim = trainer.model.classifier.output_dim # curr_entity*2 + 1 changes dynamicly
            logger.info("hidden_dim=%d, output_dim=%d"%(hidden_dim,output_dim))

        # Update the architecture of the classifier
        elif iteration == 1:
            trainer.refer_model = deepcopy(trainer.model) # old model
            trainer.refer_model.eval()
            # Change model classifier
            hidden_dim = trainer.model.classifier.hidden_dim
            output_dim = trainer.model.classifier.output_dim
            logger.info("hidden_dim=%d, old_output_dim=%d, new_output_dim=%d"%(
                                        hidden_dim,
                                        output_dim,
                                        class_per_entity*params.nb_class_pg))
            new_fc = SplitCosineLinear(hidden_dim, output_dim, class_per_entity*params.nb_class_pg)

            new_fc.fc0.weight.data = trainer.model.classifier.weight.data[:1] # for O class
            new_fc.fc1.weight.data = trainer.model.classifier.weight.data[1:] # for old class
            new_fc.sigma.data = trainer.model.classifier.sigma.data

            trainer.model.classifier = new_fc
            trainer.model.cuda()

        else:
            trainer.refer_model = deepcopy(trainer.model) # old model
            trainer.refer_model.eval()
            # Change model classifier
            hidden_dim = trainer.model.classifier.hidden_dim
            output_dim1 = trainer.model.classifier.fc1.output_dim
            output_dim2 = trainer.model.classifier.fc2.output_dim
            logger.info("hidden_dim=%d, old_output_dim=%d, new_output_dim=%d"%(
                                                            hidden_dim,
                                                            1+output_dim1+output_dim2,
                                                            class_per_entity*params.nb_class_pg))                                                
            new_fc = SplitCosineLinear(hidden_dim, 1+output_dim1+output_dim2, class_per_entity*params.nb_class_pg)

            new_fc.fc0.weight.data = trainer.model.classifier.fc0.weight.data # for O classes
            new_fc.fc1.weight.data[:output_dim1] = trainer.model.classifier.fc1.weight.data
            new_fc.fc1.weight.data[output_dim1:] = trainer.model.classifier.fc2.weight.data
            new_fc.sigma.data = trainer.model.classifier.sigma.data

            trainer.model.classifier = new_fc
            trainer.model.cuda()

        # Update entity list and label list
        if iteration==0:
            new_entity_list = ner_dataloader.entity_list[:params.nb_class_fg] # entity set of the current task/step
            all_seen_entity_list = ner_dataloader.entity_list[:params.nb_class_fg] # entity set of the up-to-now task/step
        else:
            new_entity_list = ner_dataloader.entity_list[\
                                params.nb_class_fg+(iteration-1)*params.nb_class_pg
                                :params.nb_class_fg+iteration*params.nb_class_pg] # entity set of the current task/step
            all_seen_entity_list = ner_dataloader.entity_list[\
                                :params.nb_class_fg+iteration*params.nb_class_pg] # entity set of the up-to-now task/step

        num_classes_new = 1+class_per_entity*len(all_seen_entity_list) # label numbers of the up-to-now task/step

        if iteration>0:
            num_classes_old = num_classes_new - class_per_entity*len(new_entity_list) #label numbers of the old task/step
            trainer.old_classes = num_classes_old
            trainer.nb_new_classes = class_per_entity*len(new_entity_list) 
            trainer.nb_current_classes = num_classes_new
            trainer.classes.append(trainer.nb_new_classes)
        else:
            num_classes_old = 0 #the first step/task, number of old labels is 0
            trainer.old_classes = num_classes_old
            trainer.nb_new_classes = class_per_entity*len(new_entity_list) + 1
            trainer.nb_current_classes = num_classes_new
            trainer.classes.append(trainer.nb_new_classes)


        new_classes_list = list(range(num_classes_old,num_classes_new)) # label set of the current task/step
        logger.info("All seen entity types = %s"%str(all_seen_entity_list))
        logger.info("New entity types = %s"%str(new_entity_list))
        
        # Prepare data, training and val dataset of the current task/step
        dataloader_train, dataloader_dev = ner_dataloader.get_dataloader(
                                                            first_N_classes=-1,
                                                            select_entity_list=new_entity_list,
                                                            phase=['train','dev'],
                                                            is_filter_O=params.is_filter_O,
                                                            reserved_ratio=params.reserved_ratio)
        # for debug, training and val dataset of the up-to-now task/step
        dataloader_dev_cumul, dataloader_test_cumul = ner_dataloader.get_dataloader(
                                                            first_N_classes=len(all_seen_entity_list),
                                                            select_entity_list=[],
                                                            phase=['dev','test'],
                                                            is_filter_O=False)

        if iteration > 0:
            Noise_G = Noise_G_model(params, trainer.refer_model, label_list, params.input_dim, params.num_heads, params.dropout, params.num_layers)
            Noise_G.noise_model.cuda()

        if iteration==0: # the first step/task
            # build scheduler and optimizer
            trainer.optimizer = torch.optim.SGD(trainer.model.parameters(),
                                            lr=trainer.lr,
                                            momentum=trainer.mu,
                                            weight_decay=trainer.weight_decay)

            trainer.scheduler = torch.optim.lr_scheduler.MultiStepLR(trainer.optimizer,
                                                                milestones=eval(params.schedule),
                                                                gamma=params.gamma)  

        else:
            # iteration>0
            # Update optimizer and scheduler: Fix the embedding of old classes
            # import pdb
            # pdb.set_trace()
            if params.weight_tuning:
                Lr_iteration = float(float(params.stable_lr) * math.exp(-1*params.alpha*(1+iteration)))
                Lr_iteration_noise = float(float(params.lr_noise) * math.exp(-1*params.alpha*(1+iteration)))
            else:
                Lr_iteration = float(params.stable_lr)
                Lr_iteration_noise = float(params.lr_noise)


            if params.is_fix_trained_classifier: # freeze the well trained classifier
                # if fix the O classifier
                if params.is_unfix_O_classifier: # False
                    ignored_params = list(map(id, trainer.model.classifier.fc1.parameters()))
                    base_params = filter(lambda p: id(p) not in ignored_params, \
                                trainer.model.parameters())
                    tg_params =[{'params': base_params, 'lr': Lr_iteration,
                                'weight_decay': float(params.weight_decay)}, \
                                {'params': trainer.model.classifier.fc1.parameters(), 'lr': 0., 
                                'weight_decay': 0.}]
                else:
                    ignored_params = list(map(id, trainer.model.classifier.fc1.parameters())) + \
                                    list(map(id, trainer.model.classifier.fc0.parameters()))
                    base_params = filter(lambda p: id(p) not in ignored_params, \
                                trainer.model.parameters())
                    tg_params =[{'params': base_params, 'lr': Lr_iteration,
                                'weight_decay': float(params.weight_decay)}, \
                                {'params': trainer.model.classifier.fc0.parameters(), 'lr': 0., 
                                'weight_decay': 0.}, \
                                {'params': trainer.model.classifier.fc1.parameters(), 'lr': 0., 
                                'weight_decay': 0.}]
                    base_params_niose = Noise_G.noise_model.parameters()
                    tg_params_noise =[{'params': base_params_niose, 'lr': Lr_iteration_noise,
                                'weight_decay': float(params.weight_decay)}]

            else:
                tg_params = [{'params': trainer.model.parameters(), 'lr': Lr_iteration, 
                            'weight_decay': float(params.weight_decay)}]
                tg_params_noise =[{'params': Noise_G.noise_model.parameters(), 'lr': Lr_iteration_noise,
                            'weight_decay': float(params.weight_decay)}]
            trainer.optimizer = torch.optim.SGD(tg_params, 
                                                momentum=params.mu)
            Noise_G.optimizer = torch.optim.SGD(tg_params_noise, 
                                                momentum=params.mu)
            # last_epoch_or_step = last_global_step if params.is_train_by_steps \
            #                                     else last_global_epoch
            trainer.scheduler = None
            Noise_G.scheduler = None
            # trainer.scheduler_new = WTLR(trainer.optimizer,alpha=params.WT_alpha, t=iteration)  

        # Scaling the weights in the new classifier(imprint)
        if iteration>0 and params.is_rescale_new_weight and (not params.is_from_scratch):   # True
            # (1) compute the average norm of old embdding
            old_embedding_norm = trainer.model.classifier.fc1.weight.data.norm(dim=1, keepdim=True)
            average_old_embedding_norm = torch.mean(old_embedding_norm, dim=0).cpu().type(torch.DoubleTensor)
            # (2) compute class centers for each new classes (B-/I-)
            class_center_matrix = compute_class_feature_center(dataloader_dev, 
                                        feature_model=trainer.model.encoder, 
                                        select_class_indexes=new_classes_list, 
                                        is_normalize=True,
                                        is_return_flatten_feat_and_Y=False)
            # (3) rescale the norm for each classes (each row) 
            rescale_weight_matrix = F.normalize(class_center_matrix, p=2, dim=-1) * average_old_embedding_norm
            nan_pos_list = torch.where(torch.isnan(rescale_weight_matrix[:,0]))[0]
            for nan_pos in nan_pos_list:
                assert nan_pos%2==1, "Entity not appear in dataloader!!!"
                # replace the weight of I- with B-
                rescale_weight_matrix[nan_pos] = rescale_weight_matrix[nan_pos-1].clone()
            trainer.model.classifier.fc2.weight.data = rescale_weight_matrix.type(torch.FloatTensor).cuda()

        # Init training variables
        if iteration==0 and params.first_training_epochs>0:
            training_epochs = params.first_training_epochs # 20
        else:
            training_epochs = params.training_epochs # 20

        no_improvement_num = 0
        no_improvement_num_e = 0
        best_f1 = -1
        best_f1_e = 100
        step = 0
        step_e = 0
        is_finish = False
        is_finish_e = False

        # Reset the training epoch if train by steps
        if params.is_train_by_steps: # False
            steps_per_epoch = int(len(dataloader_train.dataset)/params.batch_size)
            if iteration==0 and params.first_training_steps>0:
                training_epochs = int(params.first_training_steps/steps_per_epoch)+1
            else:
                training_epochs = int(params.training_steps/steps_per_epoch)+1

        # Check if checkpoint exists and continal training on that checkpoint
        if params.is_load_ckpt_if_exists: # True
            # skip the training of the first task/step/base
            if iteration==0 and params.is_load_common_first_model and os.path.isfile(common_first_model_ckpth_path):
                logger.info("Skip training %d-th iter checkpoint %s exists"%\
                                (iteration+1, common_first_model_ckpth_path))
                training_epochs = 0
            elif os.path.isfile(best_model_ckpt_path): # Other tasks that have been run before can also be skipped. At this time, there is no training, only testing (the previous setting is running repeatedly under the same conditions, which is equivalent to only testing)
                logger.info("Skip training %d-th iter checkpoint %s exists"%\
                                (iteration+1, best_model_ckpt_path))
                training_epochs = 0

                
        # Start training the target model
        if trainer.scheduler!=None:
            logger.info("Initial lr is %s"%( str(trainer.scheduler.get_last_lr())))

    



        ########### noise model
        if iteration>0:
            if params.AD_iter == 0 or params.AD_iter == iteration:
                # if iteration>0 and params.gaussian_noise == False:
                for e_noise in range(1, params.noise_training_epochs+1):
                    if is_finish_e:
                        break
                    logger.info("============== Noise epoch %d ==============" % e_noise)
                    # loss list: total loss, distillation loss, CE loss
                    r_loss_n, d_list_n, cs_loss_n, loss_list_n = [], [], [], []
                    # average loss
                    mean_loss_e = 0.0
                    # training acc
                    total_cnt_e, correct_cnt_e = 0, 0
                    trainer.refer_model.eval()
                    for X, y in dataloader_train:
                        # Update the step count, accumulate batch counts
                        step_e += 1
                        X, y = X.cuda(), y.cuda()
                        with torch.no_grad():
                            refer_features = trainer.refer_model.encoder.bert.embeddings(X)
                        # Forward
                        X_noise_train = Noise_G.batch_forward(refer_features)
                        # Record training accuracy
                        # pdb.set_trace()
                        mask_O_e = torch.not_equal(y, ner_dataloader.O_index) # mask 0
                        mask_pad_e = torch.not_equal(y, pad_token_label_id) # mask -100
                        eval_mask_e = torch.logical_and(mask_O_e, mask_pad_e)
                        with torch.no_grad():
                            fea = trainer.refer_model.forward_encoder(X, X_noise_train)
                            logits_e = trainer.refer_model.forward_classifier(fea)
                        predictions_e = torch.max(logits_e,dim=2)[1] # bsz * len
                        correct_cnt_e += int(torch.sum(torch.eq(predictions_e,y)[eval_mask_e].float()).item())
                        total_cnt_e += int(torch.sum(eval_mask_e.float()).item())
                        # Compute loss
                        r_loss, cs_loss, d_list = Noise_G.batch_loss_noise(y, params.cs_factor, params.l2_factor, params.d_factor)
                        r_loss_n.append(r_loss)
                        cs_loss_n.append(cs_loss)
                        d_list_n.append(d_list)
                        total_loss = Noise_G.batch_backward() # total loss
                        loss_list_n.append(total_loss) # add to the total loss of each batch
                        # pdb.set_trace()
                        mean_loss_e = np.mean(loss_list_n) # average of total loss of each batch
                        mean_r_loss = np.mean(r_loss_n) if len(r_loss_n)>0 else 0 # average of L2 loss of each batch
                        mean_cs_loss = np.mean(cs_loss_n) if len(cs_loss_n)>0 else 0 # average of CE loss of each batch
                        mean_d_loss = np.mean(d_list_n) if len(d_list_n)>0 else 0 # average of distil loss of each batch


                    # Print training information
                    if params.info_per_epochs>0 and step_e%params.info_per_epochs==0: # params.info_per_epochs=1    Output s every other epoch 
                        logger.info("Epoch %d, Step %d: Total_loss=%.3f, L2_loss=%.3f, CS_loss=%.3f, Distill_loss=%.3f, Training_exact_match=%.2f%%"%(
                                e_noise, step_e, mean_loss_e, \
                                mean_r_loss, mean_cs_loss, mean_d_loss, correct_cnt_e/total_cnt_e*100
                        ))
                    # Update lr + save skpt + do evaluation
                    # Update learning rate
                    if Noise_G.scheduler != None:
                        old_lr_e = Noise_G.scheduler.get_last_lr()
                        Noise_G.scheduler.step() # decay the LR
                        new_lr_e = Noise_G.scheduler.get_last_lr()
                        if old_lr_e != new_lr_e:
                            logger.info("Epoch %d, Step %d: lr is %s"%(
                                e_noise, step_e, str(new_lr_e)
                            ))
                    # Save checkpoint 
                    if params.save_per_epochs>0 and step_e%params.save_per_epochs==0: # params.save_per_epochs=0
                        Noise_G.save_model("noise_checkpoint_domain_%s_iteration_%d_epoch_%d.pth"%(
                                                domain_name, 
                                                iteration,
                                                e_noise), 
                                            path=params.dump_path)
                    # For evaluation
                    if not params.debug and e_noise%params.evaluate_interval==0: # params.debug=False  params.evaluate_interval=1. Evaluate every other epoch 
                        # dev/val set and entity set of current task/step
                        f1_dev_e, ma_f1_dev_e, f1_dev_each_class_e = Noise_G.evaluate(dataloader_dev_cumul, 
                                                                    each_class=True,
                                                                    entity_order=all_seen_entity_list,
                                                                    is_plot_hist=False)
                        logger.info("New data: Epoch %d, Step %d: Dev_f1=%.3f, Dev_ma_f1=%.3f, Dev_f1_each_class=%s" % (
                            e_noise, step_e, f1_dev_e, ma_f1_dev_e, str(f1_dev_each_class_e)
                        ))
                        
                        #  choose the best performance model on the current dev/val set
                        if f1_dev_e < best_f1_e: # The default is micro average, which is the preferred indicator.
                            logger.info("Find better model!!")
                            best_f1_e = f1_dev_e
                            no_improvement_num_e = 0
                            Noise_G.save_model(best_e_model_ckpt_name, path=params.dump_path)
                        else:
                            no_improvement_num_e += 1
                            logger.info("No better model is found (%d/%d)" % (no_improvement_num_e, params.early_stop))
                        if no_improvement_num_e >= params.early_stop:
                            logger.info("Stop training because no better model is found!!!")
                            is_finish_e = True
                logger.info("Finish noise training ...")
                # testing noise model
                Noise_G.load_model(best_e_model_ckpt_name, path=params.dump_path)
                Noise_G.noise_model.cuda()

                logger.info("Testing Noise Model...")

                f1_test_cumul_e, ma_f1_test_cumul_e, f1_test_each_class_cumul_e = Noise_G.evaluate(dataloader_test_cumul, 
                                                            each_class=True,
                                                            entity_order=all_seen_entity_list,
                                                            is_plot_hist=False)   
                logger.info("Accumulation: Test_f1=%.3f, Test_ma_f1=%.3f, Test_f1_each_class=%s"%(
                            f1_test_cumul_e, ma_f1_test_cumul_e, str(f1_test_each_class_cumul_e)))
                logger.info("Finish testing noise model the %d-th iter!"%(iteration+1))
            


        if iteration>0:
            trainer.before(train_loader=dataloader_train)

        for e in range(1, training_epochs+1):
            if is_finish:
                break
            logger.info("============== epoch %d ==============" % e)
            # loss list: total loss, distillation loss, CE loss
            loss_list, distill_list, ce_list = [], [], []
            # average loss
            mean_loss = 0.0
            # training acc
            total_cnt, correct_cnt = 0, 0
            

            for X, y in dataloader_train:
                if is_finish:
                    break
                # Update the step count, accumulate batch counts
                step += 1

                X, y = X.cuda(), y.cuda() # bsz*len, bsz*len, label only shown the current step entity types, others are 0 and -100
        
                # Forward
                if iteration>0 and params.gaussian_noise == False:
                    if params.AD_iter == 0 or params.AD_iter == iteration:
                        with torch.no_grad():
                            trainer.refer_model.eval()
                            Noise_G.noise_model.eval()
                            refer_features = trainer.refer_model.encoder.bert.embeddings(X)
                            X_noise = Noise_G.noise_model(refer_features)
                        trainer.batch_forward(X, X_noise) # logits.shape = bsz*len*class_num
                    else:
                        X_noise = 0
                        trainer.batch_forward(X, X_noise)
           
                        
                else:
                    X_noise = 0
                    trainer.batch_forward(X, X_noise) # logits.shape = bsz*len*class_num

                # Record training accuracy
                # pdb.set_trace()
                mask_O = torch.not_equal(y, ner_dataloader.O_index) # mask 0
                mask_pad = torch.not_equal(y, pad_token_label_id) # mask -100
                eval_mask = torch.logical_and(mask_O, mask_pad)
                predictions = torch.max(trainer.logits,dim=2)[1] # bsz * len
                correct_cnt += int(torch.sum(torch.eq(predictions,y)[eval_mask].float()).item())
                total_cnt += int(torch.sum(eval_mask.float()).item())
                # Compute loss
                if iteration>0:
                    ce_loss, distill_loss = trainer.batch_loss_cpfd(y, X_noise)
                    ce_list.append(ce_loss)
                    distill_list.append(distill_loss)
                else: #  the firt task/step only has CE loss
                    ce_loss = trainer.batch_loss(y)
                    ce_list.append(ce_loss) # loss of each batch

                total_loss = trainer.batch_backward() # total loss
                loss_list.append(total_loss) # add to the total loss of each batch
                mean_loss = np.mean(loss_list) # average of total loss of each batch
                mean_distill_loss = np.mean(distill_list) if len(distill_list)>0 else 0 # average of diatill loss of each batch
                mean_ce_loss = np.mean(ce_list) if len(ce_list)>0 else 0 # average of CE loss of each batch

                # Print training information
                if params.info_per_steps>0 and step%params.info_per_steps==0: # params.info_per_steps=0
                    logger.info("Epoch %d, Step %d: Total_loss=%.3f, CE_loss=%.3f, Distill_loss=%.3f, Training_exact_match=%.2f%%"%(
                            e, step, mean_loss, \
                            mean_ce_loss, mean_distill_loss, correct_cnt/total_cnt*100
                    ))
                    # reset the loss lst
                    loss_list = []
                    distill_list = []
                    ce_list = []
                # Update lr + save ckpt + do evaluation
                if params.is_train_by_steps: # False
                    if step>=params.training_steps:
                        is_finish = True
                    # Update learning rate
                    if trainer.scheduler != None:
                        old_lr = trainer.scheduler.get_last_lr()
                        trainer.scheduler.step()
                        new_lr = trainer.scheduler.get_last_lr()
                        if old_lr != new_lr:
                            logger.info("Epoch %d, Step %d: lr is %s"%(
                                e, step, str(new_lr)
                            ))
                    # Save checkpoint 
                    if params.save_per_steps>0 and step%params.save_per_steps==0:
                        trainer.save_model("checkpoint_domain_%s_iteration_%d_steps_%d.pth"%(
                                                domain_name, 
                                                iteration,
                                                step), 
                                            path=params.dump_path)
                    # For evaluation
                    if not params.debug and step%params.evaluate_interval==0:
                        f1_dev, ma_f1_dev, f1_dev_each_class = trainer.evaluate(dataloader_dev, 
                                                                    each_class=True,
                                                                    entity_order=new_entity_list)
                        logger.info("New data: Epoch %d, Step %d: Dev_f1=%.3f, Dev_ma_f1=%.3f, Dev_f1_each_class=%s" % (
                            e, step, f1_dev, ma_f1_dev, str(f1_dev_each_class)
                        ))
                  
                        if f1_dev > best_f1:
                            logger.info("Find better model!!")
                            best_f1 = f1_dev
                            no_improvement_num = 0
                            if iteration==0 and params.is_load_common_first_model:
                                trainer.save_model(common_first_model_ckpt_name, 
                                                    path=os.path.dirname(os.path.dirname(params.dump_path)))
                            else:
                                trainer.save_model(best_model_ckpt_name, path=params.dump_path)
                        else:
                            no_improvement_num += 1
                            logger.info("No better model is found (%d/%d)" % (no_improvement_num, params.early_stop))
                        if no_improvement_num >= params.early_stop:
                            logger.info("Stop training because no better model is found!!!")
                            is_finish = True

    

            # Print training information
            if params.info_per_epochs>0 and e%params.info_per_epochs==0: # params.info_per_epochs=1    Output s every other epoch 
                logger.info("Epoch %d, Step %d: Total_loss=%.3f, CE_loss=%.3f, Distill_loss=%.3f, Training_exact_match=%.2f%%"%(
                            e, step, mean_loss, \
                            mean_ce_loss, mean_distill_loss, correct_cnt/total_cnt*100
                    ))
            # Update lr + save skpt + do evaluation
            # Update learning rate
            if trainer.scheduler != None:
                old_lr = trainer.scheduler.get_last_lr()
                trainer.scheduler.step() # decay the LR
                new_lr = trainer.scheduler.get_last_lr()
                if old_lr != new_lr:
                    logger.info("Epoch %d, Step %d: lr is %s"%(
                        e, step, str(new_lr)
                    ))
            # Save checkpoint 
            if params.save_per_epochs>0 and e%params.save_per_epochs==0: # params.save_per_epochs=0
                trainer.save_model("checkpoint_domain_%s_iteration_%d_epoch_%d.pth"%(
                                        domain_name, 
                                        iteration,
                                        e), 
                                    path=params.dump_path)
            # For evaluation
            if not params.debug and e%params.evaluate_interval==0: # params.debug=False  params.evaluate_interval=1. Evaluate every other epoch 
                # dev/val set and entity set of current task/step
                if iteration > 0 and params.gaussian_noise == False:
                    if params.AD_iter == 0 or params.AD_iter == iteration:
                        f1_dev, ma_f1_dev, f1_dev_each_class = trainer.evaluate(dataloader_dev, 
                                                                    trainer.refer_model,
                                                                    Noise_G.noise_model,
                                                                    iteration,
                                                                    each_class=True,
                                                                    entity_order=new_entity_list)
                    else:
                        f1_dev, ma_f1_dev, f1_dev_each_class = trainer.evaluate(dataloader_dev, 
                                                                    trainer.model,
                                                                    trainer.model,
                                                                    iteration,
                                                                    each_class=True,
                                                                    entity_order=new_entity_list)
                else:
                    f1_dev, ma_f1_dev, f1_dev_each_class = trainer.evaluate(dataloader_dev, 
                                                                trainer.model,
                                                                trainer.model,
                                                                iteration,
                                                                each_class=True,
                                                                entity_order=new_entity_list)

                logger.info("New data: Epoch %d, Step %d: Dev_f1=%.3f, Dev_ma_f1=%.3f, Dev_f1_each_class=%s" % (
                    e, step, f1_dev, ma_f1_dev, str(f1_dev_each_class)
                ))
                
                #  choose the best performance model on the current dev/val set
                if f1_dev > best_f1: # The default is micro average, which is the preferred indicator.
                    logger.info("Find better model!!")
                    best_f1 = f1_dev
                    no_improvement_num = 0
                    if iteration==0 and params.is_load_common_first_model:
                        # base model
                        trainer.save_model(common_first_model_ckpt_name, 
                                            path=os.path.dirname(os.path.dirname(params.dump_path)))
                    else: # other tasks, the best model
                        trainer.save_model(best_model_ckpt_name, path=params.dump_path)
                else:
                    no_improvement_num += 1
                    logger.info("No better model is found (%d/%d)" % (no_improvement_num, params.early_stop))
                if no_improvement_num >= params.early_stop:
                    logger.info("Stop training because no better model is found!!!")
                    is_finish = True

        logger.info("Finish training ...")

        # ===========================================================================
        # testing
        if params.debug: # False
            logger.info("Skip testing for debug...")
            continue

        # load current task the best model
        if iteration==0 and params.is_load_common_first_model:
            trainer.load_model(common_first_model_ckpt_name, 
                                path=os.path.dirname(os.path.dirname(params.dump_path)))
        elif params.weight_fusion == True:
            ########### WF START
            # import pdb
            # pdb.set_trace()
            if iteration==1:
                trainer.load_model(best_model_ckpt_name, path=params.dump_path)

                if params.DBF is True:
                    merge_weight = 0.5
                else:
                    merge_weight = np.sqrt(trainer.nb_new_classes / (trainer.nb_new_classes + trainer.old_classes))

                if params.threshold_choice is True: #### further improve the performance
                    ## encoder
                    pick = torch.zeros(0).to('cuda')
                    for name, weight in trainer.refer_model.encoder.state_dict().items():
                        magnitude = abs(weight - trainer.model.encoder.state_dict()[name])
                        pick = torch.cat((pick, magnitude.flatten()), dim=0)
                    
                    merge_threshold = torch.topk(pick, int(pick.shape[0]*0.45))[0][-1]
                    print("merge_threshold is: {}".format(merge_threshold))
                    # merge_threshold = 3e-4
                    for name, weight in trainer.refer_model.encoder.state_dict().items():
                        trainer.model.encoder.state_dict()[name] = torch.where(
                        abs(weight - trainer.model.encoder.state_dict()[name]) < merge_threshold,
                        trainer.model.encoder.state_dict()[name],
                        (1 - merge_weight) * weight + merge_weight * trainer.model.encoder.state_dict()[name]
                        )

                    ## classifier
                    pick = torch.zeros(0).to('cuda')

                    magnitude = abs(trainer.refer_model.classifier.weight.data[:1] - trainer.model.classifier.fc0.weight.data)
                    pick = torch.cat((pick, magnitude.flatten()), dim=0)

                    magnitude = abs(trainer.refer_model.classifier.weight.data[1:] - trainer.model.classifier.fc1.weight.data)
                    pick = torch.cat((pick, magnitude.flatten()), dim=0)

                    magnitude = abs(trainer.refer_model.classifier.sigma.data - trainer.model.classifier.sigma.data)
                    pick = torch.cat((pick, magnitude.flatten()), dim=0)
                    
                    merge_threshold = torch.topk(pick, int(pick.shape[0]*0.45))[0][-1]
                    print("merge_threshold is: {}".format(merge_threshold))
                    # merge_threshold = 3e-4
                    trainer.model.classifier.fc0.weight.data = torch.where(
                    abs(trainer.refer_model.classifier.weight.data[:1] - trainer.model.classifier.fc0.weight.data) < merge_threshold,
                    trainer.model.classifier.fc0.weight.data,
                    (1 - merge_weight) * trainer.refer_model.classifier.weight.data[:1] + merge_weight * trainer.model.classifier.fc0.weight.data
                    )

                    trainer.model.classifier.fc1.weight.data = torch.where(
                    abs(trainer.refer_model.classifier.weight.data[1:] - trainer.model.classifier.fc1.weight.data) < merge_threshold,
                    trainer.model.classifier.fc1.weight.data,
                    (1 - merge_weight) * trainer.refer_model.classifier.weight.data[1:] + merge_weight * trainer.model.classifier.fc1.weight.data
                    )

                    trainer.model.classifier.sigma.data = torch.where(
                    abs(trainer.refer_model.classifier.sigma.data - trainer.model.classifier.sigma.data) < merge_threshold,
                    trainer.model.classifier.sigma.data,
                    (1 - merge_weight) * trainer.refer_model.classifier.sigma.data + merge_weight * trainer.model.classifier.sigma.data
                    )
                else:
                    for name, weight in trainer.refer_model.encoder.state_dict().items():
                        trainer.model.encoder.state_dict()[name] = (1 - merge_weight) * weight + merge_weight * trainer.model.encoder.state_dict()[name]

                    trainer.model.classifier.fc0.weight.data = (1 - merge_weight) * trainer.refer_model.classifier.weight.data[:1] + merge_weight * trainer.model.classifier.fc0.weight.data
                    trainer.model.classifier.fc1.weight.data = (1 - merge_weight) * trainer.refer_model.classifier.weight.data[1:] + merge_weight * trainer.model.classifier.fc1.weight.data
                    trainer.model.classifier.sigma.data = (1 - merge_weight) * trainer.refer_model.classifier.sigma.data + merge_weight * trainer.model.classifier.sigma.data

                trainer.save_model(best_model_ckpt_name, path=params.dump_path)

            else:
                trainer.load_model(best_model_ckpt_name, path=params.dump_path)

                if params.DBF is True:
                    merge_weight = 0.5
                else:
                    merge_weight = np.sqrt(trainer.nb_new_classes / (trainer.nb_new_classes + trainer.old_classes))
                
                if params.threshold_choice is True: #### further improve the performance
                    ## encoder
                    pick = torch.zeros(0).to('cuda')
                    for name, weight in trainer.refer_model.encoder.state_dict().items():
                        magnitude = abs(weight - trainer.model.encoder.state_dict()[name])
                        pick = torch.cat((pick, magnitude.flatten()), dim=0)
                    
                    merge_threshold = torch.topk(pick, int(pick.shape[0]*0.45))[0][-1]
                    print("merge_threshold is: {}".format(merge_threshold))
                    # merge_threshold = 3e-4
                    for name, weight in trainer.refer_model.encoder.state_dict().items():
                        trainer.model.encoder.state_dict()[name] = torch.where(
                        abs(weight - trainer.model.encoder.state_dict()[name]) < merge_threshold,
                        trainer.model.encoder.state_dict()[name],
                        (1 - merge_weight) * weight + merge_weight * trainer.model.encoder.state_dict()[name]
                        )

                    ## classifier
                    pick = torch.zeros(0).to('cuda')
                    
                    magnitude = abs(trainer.refer_model.classifier.fc0.weight.data - trainer.model.classifier.fc0.weight.data)
                    pick = torch.cat((pick, magnitude.flatten()), dim=0)

                    magnitude = abs(trainer.refer_model.classifier.fc1.weight.data - trainer.model.classifier.fc1.weight.data[:output_dim1])
                    pick = torch.cat((pick, magnitude.flatten()), dim=0)

                    magnitude = abs(trainer.refer_model.classifier.fc2.weight.data - trainer.model.classifier.fc1.weight.data[output_dim1:])
                    pick = torch.cat((pick, magnitude.flatten()), dim=0)

                    magnitude = abs(trainer.refer_model.classifier.sigma.data - trainer.model.classifier.sigma.data)
                    pick = torch.cat((pick, magnitude.flatten()), dim=0)
                    
                    merge_threshold = torch.topk(pick, int(pick.shape[0]*0.45))[0][-1]
                    print("merge_threshold is: {}".format(merge_threshold))
                    # merge_threshold = 3e-4
                    trainer.model.classifier.fc0.weight.data = torch.where(
                    abs(trainer.refer_model.classifier.fc0.weight.data - trainer.model.classifier.fc0.weight.data) < merge_threshold,
                    trainer.model.classifier.fc0.weight.data,
                    (1 - merge_weight) * trainer.refer_model.classifier.fc0.weight.data + merge_weight * trainer.model.classifier.fc0.weight.data
                    )

                    trainer.model.classifier.fc1.weight.data[:output_dim1] = torch.where(
                    abs(trainer.refer_model.classifier.fc1.weight.data - trainer.model.classifier.fc1.weight.data[:output_dim1]) < merge_threshold,
                    trainer.model.classifier.fc1.weight.data[:output_dim1],
                    (1 - merge_weight) * trainer.refer_model.classifier.fc1.weight.data + merge_weight * trainer.model.classifier.fc1.weight.data[:output_dim1]
                    )

                    trainer.model.classifier.fc1.weight.data[output_dim1:] = torch.where(
                    abs(trainer.refer_model.classifier.fc2.weight.data - trainer.model.classifier.fc1.weight.data[output_dim1:]) < merge_threshold,
                    trainer.model.classifier.fc1.weight.data[output_dim1:],
                    (1 - merge_weight) * trainer.refer_model.classifier.fc2.weight.data + merge_weight * trainer.model.classifier.fc1.weight.data[output_dim1:]
                    )

                    trainer.model.classifier.sigma.data = torch.where(
                    abs(trainer.refer_model.classifier.sigma.data - trainer.model.classifier.sigma.data) < merge_threshold,
                    trainer.model.classifier.sigma.data,
                    (1 - merge_weight) * trainer.refer_model.classifier.sigma.data + merge_weight * trainer.model.classifier.sigma.data
                    )
                else:
                    for name, weight in trainer.refer_model.encoder.state_dict().items():
                        trainer.model.encoder.state_dict()[name] = (1 - merge_weight) * weight + merge_weight * trainer.model.encoder.state_dict()[name]

                    trainer.model.classifier.fc0.weight.data = (1 - merge_weight) * trainer.refer_model.classifier.fc0.weight.data + merge_weight * trainer.model.classifier.fc0.weight.data
                    trainer.model.classifier.fc1.weight.data[:output_dim1] = (1 - merge_weight) * trainer.refer_model.classifier.fc1.weight.data + merge_weight * trainer.model.classifier.fc1.weight.data[:output_dim1]
                    trainer.model.classifier.fc1.weight.data[output_dim1:] = (1 - merge_weight) * trainer.refer_model.classifier.fc2.weight.data + merge_weight * trainer.model.classifier.fc1.weight.data[output_dim1:]
                    trainer.model.classifier.sigma.data = (1 - merge_weight) * trainer.refer_model.classifier.sigma.data + merge_weight * trainer.model.classifier.sigma.data

                trainer.save_model(best_model_ckpt_name, path=params.dump_path)

            ########### WF END
        else:
            trainer.load_model(best_model_ckpt_name, path=params.dump_path)
        trainer.model.cuda()


        # testing    test set and entity set of the up-to-now task/step
        logger.info("Testing...")
        # pdb.set_trace()

        if iteration > 0 and params.gaussian_noise == False:
            if params.AD_iter == 0 or params.AD_iter == iteration:
                f1_test_cumul, ma_f1_test_cumul, f1_test_each_class_cumul = trainer.evaluate(dataloader_test_cumul, 
                                                            trainer.refer_model,
                                                            Noise_G.noise_model,
                                                            iteration,
                                                            each_class=True,
                                                            entity_order=all_seen_entity_list,
                                                            is_plot_hist=False)  
            else: 
                f1_test_cumul, ma_f1_test_cumul, f1_test_each_class_cumul = trainer.evaluate(dataloader_test_cumul, 
                                                            trainer.model,
                                                            trainer.model,
                                                            iteration,
                                                            each_class=True,
                                                            entity_order=all_seen_entity_list,
                                                            is_plot_hist=False) 
        else: 
            f1_test_cumul, ma_f1_test_cumul, f1_test_each_class_cumul = trainer.evaluate(dataloader_test_cumul, 
                                                        trainer.model,
                                                        trainer.model,
                                                        iteration,
                                                        each_class=True,
                                                        entity_order=all_seen_entity_list,
                                                        is_plot_hist=False)  
        logger.info("Accumulation: Test_f1=%.3f, Test_ma_f1=%.3f, Test_f1_each_class=%s"%(
                    f1_test_cumul, ma_f1_test_cumul, str(f1_test_each_class_cumul)))
        logger.info("Finish testing the %d-th iter!"%(iteration+1))

        

if __name__ == "__main__":
    params = get_params() # get configs
    main_cl(params)

    
