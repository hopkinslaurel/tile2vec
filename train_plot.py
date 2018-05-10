import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_loss(epoch, train_loss, test_loss, lsms_loss, fname):
    plt.figure()
    plt.plot(list(range(0,epoch+1)), train_loss, 'b', label='train')
    plt.plot(list(range(0,epoch+1)), test_loss, 'r', label='test')
    plt.plot(list(range(0,epoch+1)), lsms_loss, 'g', label='lsms')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend()
    plt.show()
    plt.savefig(fname)
    plt.close()

def plot_r2_mse(epoch, r2_list, mse_list, fname):
    plt.figure()
    plt.plot(list(range(0,epoch+1)), r2_list, 'b', label='r^2')
    plt.plot(list(range(0,epoch+1)), mse_list, 'r', label='mse')
    plt.xlabel('Epoch', fontsize=14)
    plt.legend()
    plt.show()
    plt.savefig(fname)
    plt.close()
