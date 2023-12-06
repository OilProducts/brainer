import time
import torch

def train_hebb(model, loader, device, measures=None, criterion=None):
    """
    Train only the hebbian blocks
    """
    t = time.time()
    loss_acc = (not model.is_hebbian()) and (criterion is not None)
    with torch.no_grad():
        for inputs, target in loader:
            # print(inputs.min(), inputs.max(), inputs.mean(), inputs.std())
            ## 1. forward propagation
            inputs = inputs.float().to(device)  # , non_blocking=True)
            output = model(inputs)

            # print(r"%s"%(time.time()-t))

            if loss_acc:
                target = target.to(device, non_blocking=True)

                ## 2. loss calculation
                loss = criterion(output, target)

                ## 3. Accuracy assessment
                predict = output.data.max(1)[1]
                acc = predict.eq(target.data).sum()
                # Save if measurement is wanted
                conv, r1 = model.convergence()
                measures.step(target.shape[0], loss.clone().detach().cpu(), acc.cpu(), conv, r1, model.get_lr())
            model.update()

    info = model.radius()
    convergence, R1 = model.convergence()
    return measures, model.get_lr(), info, convergence, R1

def train_unsup(model, loader, device,
                blocks=[]):  # fixed bug as optimizer is not used or pass in the only use it has in this repo currently
    """
    Unsupervised learning only works with hebbian learning
    """
    model.train(blocks=blocks)  # set unsup blocks to train mode
    _, lr, info, convergence, R1 = train_hebb(model, loader, device)
    return lr, info, convergence, R1