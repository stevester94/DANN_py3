import torch.backends.cudnn as cudnn
import torch.utils.data

def test(model, label_loss_func, domain_loss_func, iterator):
    cuda = True
    cudnn.benchmark = True
    alpha = 0

    model = model.eval()

    n_batches = 0
    n_total = 0
    n_correct = 0
    total_label_loss = 0
    total_domain_loss = 0

    for x,y,t in iterator:

        # test model using target data
        # x = torch.from_numpy(x)
        # y = torch.from_numpy(y).long()
        # t = torch.from_numpy(t).long()

        batch_size = len(t)

        if cuda:
            model = model.cuda()
            x = x.cuda()
            y = y.cuda()
            t = t.cuda()

        y_hat, t_hat = model(x=x, t=t, alpha=alpha)
        t_hat = torch.flatten(t_hat)

        pred = y_hat.data.max(1, keepdim=True)[1]

        n_correct += pred.eq(y.data.view_as(pred)).cpu().sum()
        n_total += batch_size
        total_label_loss += label_loss_func(y_hat, y).cpu().item()
        total_domain_loss += domain_loss_func(t_hat, t).cpu().item()

        n_batches += 1

    accu = n_correct.data.numpy() * 1.0 / n_total
    average_label_loss = total_label_loss / n_batches
    average_domain_loss = total_domain_loss / n_batches

    return accu, average_label_loss, average_domain_loss
