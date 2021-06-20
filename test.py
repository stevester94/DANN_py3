import torch.backends.cudnn as cudnn
import torch.utils.data

def test(model, np_iterator):
    cuda = True
    cudnn.benchmark = True
    alpha = 0

    model = model.eval()


    if cuda:
        model = model.cuda()

    i = 0
    n_total = 0
    n_correct = 0

    for x,y,t in np_iterator:

        # test model using target data
        x = torch.from_numpy(x)
        y = torch.from_numpy(y).long()
        t = torch.from_numpy(t).long()

        batch_size = len(t)

        if cuda:
            x = x.cuda()
            y = y.cuda()
            t = t.cuda()

        y_hat, _ = model(input_data=x, alpha=alpha)
        pred = y_hat.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    return accu
