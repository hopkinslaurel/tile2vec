import numpy as np
from time import time
from torch.autograd import Variable
import torch
from src.datasets import triplet_dataloader
import sys

# https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def prep_triplets(triplets, cuda):
    """
    Takes a batch of triplets and converts them into Pytorch variables 
    and puts them on GPU if available.
    """
    #print('prep_triplets')
    a, n, d = (Variable(triplets['anchor']), Variable(triplets['neighbor']), Variable(triplets['distant']))
    idx = triplets['idx'].numpy()
    #print(idx)  
    #print("prep_triplets")
    #print(a.size())
    if cuda:
    	a, n, d = (a.cuda(), n.cuda(), d.cuda())
    return (a, n, d, idx)

def train_model(model, cuda, dataloader, optimizer, epoch, species, csv_writer, csv_writer_indv, 
    margin=1, l2=0, print_every=100, t0=None):
    """
    Trains a model for one epoch using the provided dataloader.
    """
    model.train()
    if t0 is None:
        t0 = time.time()
    sum_loss, sum_l_n, sum_l_d, sum_l_nd = (0, 0, 0, 0)
    n_train, n_batches = len(dataloader.dataset), len(dataloader)
    #print("In train, n_train: " + str(n_train))
    #print("In train, n_batchs: " + str(n_batches))
    print_sum_loss = 0
    for idx, triplets in enumerate(dataloader):
        p, n, d, triplet_idx = prep_triplets(triplets, cuda)
        optimizer.zero_grad()
        writer = None
        if idx+1 == 1 or idx+1 == n_batches:  # 3125 = 300k images / 96 imgs/batch
            writer = csv_writer_indv
        loss, l_n, l_d, l_nd = model.loss(p, n, d, triplet_idx, species, csv_writer_indv, epoch, idx+1, margin=margin, l2=l2) 
        #print("loss: ")
        #print(loss, l_n, l_d, l_nd)
        #print(loss.item(), l_n.item(), l_d.item(), l_nd.item())
        #csv_writer.writerow([idx])
        csv_writer.writerow([loss.item(), l_n.item(), (-1)*l_d.item(), l_nd.item()])
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()  #data[0]
        sum_l_n += l_n.item()  #data[0]
        sum_l_d += l_d.item()  #data[0]
        sum_l_nd += l_nd.item()  #data[0]
        if (idx + 1) * dataloader.batch_size % print_every == 0:
            print_avg_loss = (sum_loss - print_sum_loss) / (
                print_every / dataloader.batch_size)
            print('Epoch {}: [{}/{} ({:0.0f}%)], Avg loss: {:0.4f}'.format(
                epoch, (idx + 1) * dataloader.batch_size, n_train,
                100 * (idx + 1) / n_batches, print_avg_loss))
            print_sum_loss = sum_loss
    avg_loss = sum_loss / n_batches
    avg_l_n = sum_l_n / n_batches
    avg_l_d = sum_l_d / n_batches
    avg_l_nd = sum_l_nd / n_batches
    #csv_writer_avg.writerow([avg_loss, avg_l_n, (-1)*avg_l_d, avg_l_nd])
    #print('Finished epoch {}: {:0.3f}s'.format(epoch, time()-t0))
    print('\nTrain Epoch {}: Loss {:0.4f}, Time {:0.3f}s'.format(epoch, avg_loss, time()-t0))
    #print('  Average loss: {:0.4f}'.format(avg_loss))
    #print('  Average l_n: {:0.4f}'.format(avg_l_n))
    #print('  Average l_d: {:0.4f}'.format(avg_l_d))
    #print('  Average l_nd: {:0.4f}\n'.format(avg_l_nd))
    return avg_loss #(avg_loss, avg_l_n, avg_l_d, avg_l_nd)

def validate_model(model, cuda, dataloader, optimizer, epoch, species, csv_writer,  
    margin=1, l2=0, print_every=100, t0=None):
    """
    Validates a model using the provided dataloader.
    """
    
    with torch.no_grad():   # added per https://discuss.pytorch.org/t/out-of-memory-error-during-evaluation-but-training-works-fine/12274/4
        model.eval()
        if t0 is None:
            t0 = time.time()
        sum_loss, sum_l_n, sum_l_d, sum_l_nd = (0, 0, 0, 0)
        n_train, n_batches = len(dataloader.dataset), len(dataloader)
        print_sum_loss = 0
        for idx, triplets in enumerate(dataloader):
            p, n, d, triplet_idx = prep_triplets(triplets, cuda)
            '''
            if idx+1 == 1 or idx+1 == 3:  # 3125 = 300k images / 96 imgs/batch -1 
                print('Epoch: ' + str(epoch))
                print('Idx: ' + str(idx))
                csv_writer_indv.writerow([epoch, idx+1])
                loss, l_n, l_d, l_nd = model.loss_write(p, n, d, csv_writers_indv, margin=margin, l2=l2)
            '''
            loss, l_n, l_d, l_nd = model.loss(p, n, d, triplet_idx, species, None, epoch, idx+1, margin=margin, l2=l2)
            #print("loss: ")
            #print(loss, l_n, l_d, l_nd)
            #print(loss.item(), l_n.item(), l_d.item(), l_nd.item())
            #writer.writerow([idx])
            csv_writer.writerow([loss.item(), l_n.item(), l_d.item(), l_nd.item()])
            sum_loss += loss.item()  #data[0]
            sum_l_n += l_n.item()  #data[0]
            sum_l_d += l_d.item()  #data[0]
            sum_l_nd += l_nd.item()  #data[0]
        avg_loss = sum_loss / n_batches
        avg_l_n = sum_l_n / n_batches
        avg_l_d = sum_l_d / n_batches
        avg_l_nd = sum_l_nd / n_batches
        #csv_writer_avg.writerow([avg_loss, avg_l_n, (-1)*avg_l_d, avg_l_nd])
        #print('Finished epoch {}: {:0.3f}s'.format(epoch, time()-t0))
        #print('  Average loss: {:0.4f}'.format(avg_loss))
        #print('  Average l_n: {:0.4f}'.format(avg_l_n))
        #print('  Average l_d: {:0.4f}'.format(avg_l_d))
        #print('  Average l_nd: {:0.4f}\n'.format(avg_l_nd))
        print('Test Epoch {}: Loss {:0.4f}, Time {:0.3f}s'.format(epoch, avg_loss, time()-t0))
    return avg_loss #(avg_loss, avg_l_n, avg_l_d, avg_l_nd)
