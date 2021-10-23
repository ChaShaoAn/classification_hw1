from matplotlib import pyplot as plt


def update_lost_hist(train_list, val_list, name='compare', xlabel='Loss'):
    plt.plot(train_list)
    plt.plot(val_list)
    plt.title(name)
    plt.ylabel(xlabel)
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='center right')
    plt.show()


def update_lost_hist_only_train(train_list, name='compare'):
    plt.plot(train_list)
    plt.title(name)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train'], loc='center right')
    plt.savefig('output/loss_compare.jpg')
    plt.show()
