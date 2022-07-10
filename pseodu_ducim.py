from data_loader import Dataloader
# from run import Run
from models import Models
from run import *
from sklearn.metrics import f1_score, make_scorer
from data_loader import TorchDataset
# from deep import AE
import torch.nn as nn
import torch
import torch
# from deep import CNN
# from data_loader import TorchDataset
import time
from models import TruncatedCNN
from models import AdverserailCNN
import seaborn as sns
import sklearn

def calculate_accuracy(model, dataloader, device):
    model.eval() # put in evaluation mode
    total_correct = 0
    total_rr = 0
    confusion_matrix = np.zeros([len(dataloader.dataset.hash_id), len(dataloader.dataset.hash_id)], int)
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_rr += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1

    model_accuracy = total_correct / total_rr * 100
    return model_accuracy, confusion_matrix


def cosine_loss(out1, out2, lbl1, lbl2, lmbda=0.5):
    cos = nn.CosineSimilarity()
    res1 = torch.abs(cos(out1, out2))  # , torch.abs(cos(out1, out2))
    res2 = torch.abs(1 - cos(out1, out2))
    res = torch.cat((torch.unsqueeze(res1, 0), torch.unsqueeze(res2, 0)), 0)
    res = res.t()
    lbl = 1*(lbl1 == lbl2)
    nll_loss = nn.CrossEntropyLoss()
    L1_loss = nn.L1Loss()
    loss = nll_loss(res, lbl) + lmbda*L1_loss(res1, res2)
    # loss = cross_entropy(torch.tensor(res2, dtype=float, requires_grad=True), torch.tensor(lbl, dtype=torch.double))
    # lbl = torch.tensor([1*indicator, 1-1*indicator])
    return loss


if __name__ == '__main__':
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    learning_rate = 1e-4
    # batch_size = 3500  # 1000 works well for 5 ids with both med and no med

    data_loader = Dataloader(input_type='raw', dataset_name=250)
    data_loader.load()
    data_loader.split()
    data_loader.clean()
    label_dict = {'k_id': 2, 'med': [0], 'age': [6]}

    data_loader.choose_specific_xy(label_dict=label_dict)
    ds_train = TorchDataset(data_loader, 'train')
    batch_size_train = int(np.ceil(0.15*ds_train.X.shape[1]))
    if batch_size_train <= 15:
        batch_size_train = int(np.ceil(0.5*ds_train.X.shape[1]))
    ds_test = TorchDataset(data_loader, 'test')
    batch_size_test = int(np.ceil(0.15*ds_test.X.shape[1]))
    d = {}
    model_obj = Models(data_loader, model_name='CNN', mode='train', **d)
    if model_obj.model_name not in ['log_reg', 'svm', 'rfc', 'xgb'] and data_loader.input_type == 'features':
        raise Exception('Only raw data should be fed to a neural network!')
    model = AdverserailCNN(250)
    model = model.to(device)
    model2 = AdverserailCNN(250)
    model2 = model2.to(device)
    kldiv = nn.KLDivLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)
    epochs = 600
    batch_acc = []
    for epoch in range(1, epochs + 1):
        model.train()  # put in training mode
        model2.train()
        running_loss = 0.0
        epoch_time = time.time()
        trainloader1 = torch.utils.data.DataLoader(
            ds_train, batch_size=batch_size_train, shuffle=True)
        trainloader2 = torch.utils.data.DataLoader(
            ds_train, batch_size=batch_size_train, shuffle=False)
        testloader1 = torch.utils.data.DataLoader(
            ds_test, batch_size=batch_size_test, shuffle=True)
        testloader2 = torch.utils.data.DataLoader(
            ds_test, batch_size=batch_size_test, shuffle=False)

        for i, data in enumerate(zip(trainloader1, trainloader2), 0):
            # get the inputs
            inputs1, labels1 = data[0]
            inputs2, labels2 = data[1]
            # send them to device
            inputs1 = inputs1.to(device)
            labels1 = labels1.to(device)
            inputs2 = inputs2.to(device)
            labels2 = labels2.to(device)

            # forward + backward + optimize
            outputs1 = model(inputs1)  # forward pass
            outputs2 = model2(inputs2)
            loss = cosine_loss(outputs1, outputs2, labels1, labels2, lmbda=0)
            # pp = 1*(labels1 == labels2)
            # if torch.sum(pp) == 0:
            #     print('No matching labels')
            # else:
            #     print('ratio of matching labels is: {:.3f} %'.format((100/len(pp))*torch.sum(pp)))
            # loss = kldiv(outputs1, outputs2) + kldiv(outputs2, outputs1) # calculate the loss
            # always the same 3 steps
            optimizer.zero_grad()  # zero the parameter gradients
            optimizer2.zero_grad()

            loss.backward()  # backpropagation
            optimizer.step()  # update parameters
            optimizer2.step()

            # print statistics
            running_loss += loss.data.item()

        # Normalizing the loss by the total number of train batches

        running_loss /= len(trainloader1)

        # Calculate training/test set accuracy of the existing model
        # train_accuracy, _ = calculate_accuracy(model, trainloader1, device)
        # test_accuracy, _ = calculate_accuracy(model, testloader, device)

        # log = "Epoch: {} | Loss: {:.4f} | Training accuracy: {:.3f}% | Test accuracy: {:.3f}% | ".format(epoch, running_loss, train_accuracy, test_accuracy)
        log = "Epoch: {} | Loss: {:.4f}  | ".format(epoch, running_loss)
        epoch_time = time.time() - epoch_time
        log += "Epoch Time: {:.2f} secs".format(epoch_time)
        print(log)
        print("Averaged gradients in first layer of model and model 2 are {:.3f} and {:.3f}".format(torch.mean(torch.abs(model.conv1[0].weight.grad)), torch.mean(torch.abs(model2.conv1[0].weight.grad))))


        # save model
        if epoch % 20 == 0:
            print('==> Saving model ...')
            state = {
                'net': model.state_dict(),
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoints_ducim'):
                os.mkdir('checkpoints_ducim')
            torch.save(state, './checkpoints_ducim/ducim.pth')

    print('==> Finished Training ...')



    # load model, calculate accuracy and confusion matrix
    # model = CifarCNN().to(device)
    # model = model.to(device)
    # state = torch.load('./checkpoints/ducim.pth', map_location=device)
    # model.load_state_dict(state['net'])
    model.eval()
    model2.eval()
    # truncated_model = TruncatedCNN(list(model.children()))
    # truncated_model = nn.Sequential(*list(model.children())[0:-1])
    correct_pred = 0
    total = 0
    for i, data in enumerate(zip(trainloader1, trainloader2), 0):
        # get the inputs
        inputs1, labels1 = data[0]
        inputs2, labels2 = data[1]
        # send them to device
        inputs1 = inputs1.to(device)
        labels1 = labels1.to(device)
        inputs2 = inputs2.to(device)
        labels2 = labels2.to(device)

        # forward + backward + optimize
        outputs1 = model(inputs1)  # forward pass
        outputs2 = model2(inputs2)
        cos = nn.CosineSimilarity()
        res2 = torch.abs(1 - cos(outputs1, outputs2))
        res_gt = 1*(labels1 == labels2)
        # xx = []
        # for x in labels1.data:
        #     xx.append(x.item())
        # # sns.heatmap(res_cpu.detach().numpy(), xticklabels=xx, yticklabels=xx, vmin=0, vmax=1)
        # # rs = sklearn.metrics.pairwise.cosine_distances(torch.Tensor.cpu(outputs1).detach().numpy())
        # # ww = np.abs(1-rs)
        # # ww[ww < 0.75] = 0
        #
        #
        # q = np.array([xx])  # make it 2D
        # gt = (q == q.T)
        # sns.heatmap(res2.cpu().detach().numpy(), xticklabels=xx, yticklabels=xx, linewidths=.5)
        # sns.heatmap(gt.cpu().detach().numpy(), xticklabels=xx, yticklabels=xx, linewidths=.5)
        # plt.show()
        res2[res2 > 0.55] = 1
        correct_pred += torch.sum(1*(res2 == res_gt))
        total += len(res2)
        a=1
    # test_accuracy, confusion_matrix = calculate_accuracy(model, testloader, device)

    print('accuracy is {:.3f} %'.format(100*correct_pred/total))
    print("test accuracy: {:.3f}%".format(test_accuracy))

    # plot confusion matrix
    # fig, ax = plt.subplots(1,1,figsize=(8,6))
    # ax.matshow(confusion_matrix, aspect='auto', vmin=0, vmax=1000, cmap=plt.get_cmap('Blues'))
    # plt.ylabel('Actual Category')
    # plt.yticks(range(10), classes)
    # plt.xlabel('Predicted Category')
    # plt.xticks(range(10), classes)
    # plt.show()


    a = 1