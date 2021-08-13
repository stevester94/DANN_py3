import matplotlib.pyplot as plt

def _do_loss_curve(history):
    figure, axis = plt.subplots(2, 1)

    figure.set_size_inches(12, 12)
    figure.suptitle("Loss During Training")
    plt.subplots_adjust(hspace=0.4)
    plt.rcParams['figure.dpi'] = 600
    
    axis[0].set_title("Label Loss")
    axis[0].plot(history["indices"], history['source_val_label_loss'], label='Source Validation Label Loss')
    axis[0].plot(history["indices"], history['source_train_label_loss'], label='Source Train Label Loss')
    axis[0].plot(history["indices"], history['target_val_label_loss'], label='Target Validation Label Loss')
    axis[0].legend()
    axis[0].grid()
    axis[0].set(xlabel='Epoch', ylabel="CrossEntropy Loss")
    axis[0].locator_params(axis="x", integer=True, tight=True)
    
    # axis[0].xlabel('Epoch')

    axis[1].set_title("Domain Loss")
    axis[1].plot(history["indices"], history['target_val_domain_loss'], label='Source Validation Domain Loss')
    axis[1].plot(history["indices"], history['source_train_domain_loss'], label='Source Train Domain Loss')
    axis[1].plot(history["indices"], history['source_val_domain_loss'], label='Target Validation Domain Loss')
    axis[1].legend()
    axis[1].grid()
    axis[1].set(xlabel='Epoch', ylabel="L1 Loss")
    axis[1].locator_params(axis="x", integer=True, tight=True)


def plot_loss_curve(history):
    _do_loss_curve(history)
    plt.show()

def save_loss_curve(history, path):
    _do_loss_curve(history)
    plt.savefig(path)