import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to

import warnings

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts, measures=True, plot=False):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts, measures, plot)
    #cost = rollout(model, dataset, opts, False, False)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, opts, measures=False, plot_data=False):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            #cost, _, _, _, _, _ = model(move_to(bat, opts.device))
            #cost, _, misMatch_y, misMatch_x, recall_accuracy, radius_mean, radius_max, recall_accuracy_max, recall_accuracy_min = model(move_to(bat[0], opts.device), labels=move_to(bat[1], opts.device), Measures=measures, Plot_Data=plot_data)
            cost, _, misMatch, _, recall_accuracy, radius_mean, radius_max, recall_accuracy_max, recall_accuracy_min = model(move_to(bat[0], opts.device), move_to(bat[1], opts.device), opts, Training=False, Measures=measures, Plot_Data=plot_data)
            #cost, _, misMatch, _, recall_accuracy, radius_mean, radius_max, recall_accuracy_max, recall_accuracy_min = model(move_to(bat, opts.device), Measures=measures, Plot_Data=plot_data)
        #return cost.data.cpu(), misMatch_y, misMatch_x, recall_accuracy, radius_mean, radius_max, recall_accuracy_max, recall_accuracy_min 
        return cost.data.cpu(), misMatch, None, recall_accuracy, radius_mean, radius_max, recall_accuracy_max, recall_accuracy_min 

    if not measures:
        return torch.cat([
            eval_model_bat(bat)[0]
            for bat
            in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
        ], 0)
    else:
        count = 0
        cost_all = torch.tensor([]).data.cpu()
        #misMatch_y_all, misMatch_x_all, recall_accuracy_all, radius_mean_all, radius_max_all = [], [], [], [], []
        #misMatch_y_all, misMatch_x_all, recall_accuracy_all, radius_mean_all = 0., 0., 0., 0.
        misMatch_all, recall_accuracy_all, radius_mean_all = 0., 0., 0.
        radius_max = torch.FloatTensor([0.]).cuda()
        recall_accuracy_max = torch.FloatTensor([0.]).cuda()
        recall_accuracy_min = torch.FloatTensor([1.]).cuda()       
        """
        radius_max = 0
        recall_accuracy_max = 0.
        recall_accuracy_min = 1.       
        """
        #misMatch_all = []
        for bat in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar):
            #cost_batch, misMatch_y_b, misMatch_x_b, recall_accuracy_b, radius_mean_b, radius_max_b, recall_accuracy_max_b, recall_accuracy_min_b = eval_model_bat(bat)
            cost_batch, misMatch_b, _, recall_accuracy_b, radius_mean_b, radius_max_b, recall_accuracy_max_b, recall_accuracy_min_b = eval_model_bat(bat)
            #cost_batch, misMatch_y_b, misMatch_x_b, _, _, _ = eval_model_bat(bat)
            #if first_iteration:
            if count == 0:
                cost_all = cost_batch
            else:
                cost_all = torch.cat((cost_all, cost_batch), 0)
            #misMatch_all += misMatch_b

            #misMatch_y_all += misMatch_y_b
            #misMatch_x_all += misMatch_x_b
            misMatch_all += misMatch_b
            recall_accuracy_all += recall_accuracy_b
            #radius_mean_all += radius_mean_b
            
            recall_accuracy_max = torch.max(recall_accuracy_max, recall_accuracy_max_b)
            recall_accuracy_min = torch.min(recall_accuracy_min, recall_accuracy_min_b)
            #radius_max = torch.max(radius_max, radius_max_b)

            #recall_accuracy_all.append(recall_accuracy_b)
            #radius_mean_all.append(radius_mean_b)
            #radius_max_all.append(radius_max_b)
            #recall_accuracy_all += recall_accuracy_b
            #recall_accuracy_max = max(recall_accuracy_max_b, recall_accuracy_max)
            #recall_accuracy_min = min(recall_accuracy_min_b, recall_accuracy_min)
            #radius_mean_all += radius_mean_b
            #radius_max = max(radius_max, radius_max_b)
            count += 1
        #print("Validation count of misMatch on y axis: {:.2f}".format((misMatch_y_all/count).item()))
        #print("Validation count of misMatch on x axis: {:.2f}".format((misMatch_x_all/count).item()))
        print("Validation count of misMatch: {:.2f}".format((misMatch_all/count).item()))
        print("Validation mean of recall_accuracy: {:.2f}".format((recall_accuracy_all/count).item()))
        print("Validation max of recall_accuracy: {:.2f}".format((recall_accuracy_max).item()))
        print("Validation min of recall_accuracy: {:.2f}".format((recall_accuracy_min).item()))
        #print("Validation mean of radius_misposition: {:.2f}".format((radius_mean_all/count).item()))
        #print("Validation max of radius_misposition: {:d}".format(round(radius_max.item())))

        #print("Validation count of misMatch on y axis: ", misMatch_y_all/count)
        #print("Validation count of misMatch on x axis: ", misMatch_x_all/count)
        #print("Validation mean of recall_accuracy: {:.2f}".format(recall_accuracy_all/count))
        #print("Validation max of recall_accuracy: {:.2f}".format(recall_accuracy_max))
        #print("Validation min of recall_accuracy: {:.2f}".format(recall_accuracy_min))
        #print("Validation mean of radius_misposition: {:.2f}".format(radius_mean_all/count))
        #print("Validation max of radius_misposition: {:d}".format(round(radius_max)))
        #print("Validation min of recall_accuracy: {:.2f}".format(recall_accuracy_min*100))
        #print("Validation mean of recall_accuracy: {:.2f}".format(sum(recall_accuracy_all)*100/len(recall_accuracy_all)))
        #print("Validation mean of radius_misposition: {:.2f}".format(sum(radius_mean_all)/len(radius_mean_all)))
        #radius_max_all.sort()
        #print("Validation max of radius_misposition: {:d}".format(radius_max_all[-1]))
        return cost_all
           
def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, train_dataset, problem, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    #training_dataset = baseline.wrap_dataset(problem.make_dataset(
    #    size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))
    training_dataset = baseline.wrap_dataset(train_dataset)
    #training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )
    avg_reward = validate(model, val_dataset, opts)

    """
    if epoch == opts.n_epochs-1:
        #avg_reward = validate(model, val_dataset, opts, plot=True)
        avg_reward = validate(model, val_dataset, opts)
    else:
        avg_reward = validate(model, val_dataset, opts)
    """
        
    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    
    x, bl_val = baseline.unwrap_batch(batch)
    #x = move_to(x, opts.device)
    x[0] = move_to(x[0], opts.device)
    x[1] = move_to(x[1], opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood, _, _, _, _, _, _, _ = model(x[0], x[1], opts, epoch)
    #cost, log_likelihood, _, _, _, _, _, _, _ = model(x)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)
