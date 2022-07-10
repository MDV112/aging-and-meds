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


if __name__ == '__main__':
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    learning_rate = 1e-4
    # batch_size = 3500  # 1000 works well for 5 ids with both med and no med

    data_loader = Dataloader(input_type='raw', dataset_name=250)
    data_loader.load()
    data_loader.split()
    data_loader.clean()
    label_dict = {'k_id': 'all', 'med': 'all', 'age': 'all'}

    data_loader.choose_specific_xy_rr(label_dict=label_dict)
    ds_train = TorchDataset(data_loader, 'train')
    batch_size_train = int(np.ceil(0.15*ds_train.X.shape[1]))
    ds_test = TorchDataset(data_loader, 'test')
    batch_size_test = int(np.ceil(0.15*ds_test.X.shape[1]))
    d = {}
    model_obj = Models(data_loader, model_name='CNN', mode='train', **d)
    if model_obj.model_name not in ['log_reg', 'svm', 'rfc', 'xgb'] and data_loader.input_type == 'features':
        raise Exception('Only raw data should be fed to a neural network!')
    model = model_obj.set_model()
    model = model.to(device)
    model2 = model_obj.set_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 200
    batch_acc = []
    for epoch in range(1, epochs + 1):
        model.train()  # put in training mode
        running_loss = 0.0
        epoch_time = time.time()
        trainloader = torch.utils.data.DataLoader(
            ds_train, batch_size=batch_size_train, shuffle=True)
        testloader = torch.utils.data.DataLoader(
            ds_test, batch_size=batch_size_test, shuffle=True)

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # send them to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            outputs = model(inputs)  # forward pass
            loss = criterion(outputs, labels)  # calculate the loss
            # always the same 3 steps
            optimizer.zero_grad()  # zero the parameter gradients
            loss.backward()  # backpropagation
            optimizer.step()  # update parameters

            # print statistics
            running_loss += loss.data.item()

        # Normalizing the loss by the total number of train batches
        running_loss /= len(trainloader)

        # Calculate training/test set accuracy of the existing model
        train_accuracy, _ = calculate_accuracy(model, trainloader, device)
        # test_accuracy, _ = calculate_accuracy(model, testloader, device)

        # log = "Epoch: {} | Loss: {:.4f} | Training accuracy: {:.3f}% | Test accuracy: {:.3f}% | ".format(epoch, running_loss, train_accuracy, test_accuracy)
        log = "Epoch: {} | Loss: {:.4f} | Training accuracy: {:.3f}% | ".format(epoch, running_loss, train_accuracy)
        epoch_time = time.time() - epoch_time
        log += "Epoch Time: {:.2f} secs".format(epoch_time)
        print(log)
        batch_acc.append(train_accuracy)

        # save model
        if epoch % 20 == 0:
            print('==> Saving model ...')
            state = {
                'net': model.state_dict(),
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoints'):
                os.mkdir('checkpoints')
            torch.save(state, './checkpoints/cifar_cnn_ckpt.pth')

    print('==> Finished Training ...')



    # load model, calculate accuracy and confusion matrix
    # model = CifarCNN().to(device)
    model = model.to(device)
    state = torch.load('./checkpoints/cifar_cnn_ckpt.pth', map_location=device)
    model.load_state_dict(state['net'])
    truncated_model = TruncatedCNN(list(model.children()))
    # truncated_model = nn.Sequential(*list(model.children())[0:-1])
    for i, data in enumerate(testloader, 0):
        # get the inputs
        inputs, labels = data
        # send them to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward + backward + optimize
        outputs = truncated_model(inputs)
        res = outputs@outputs.t()
        res = (1/torch.max(res).item())*res
        res_cpu = torch.Tensor.cpu(res)
        xx = []
        for x in labels.data:
            xx.append(x.item())
        # sns.heatmap(res_cpu.detach().numpy(), xticklabels=xx, yticklabels=xx, vmin=0, vmax=1)
        rs = sklearn.metrics.pairwise.cosine_distances(torch.Tensor.cpu(outputs).detach().numpy())
        ww = np.abs(1-rs)
        ww[ww < 0.75] = 0
        sns.heatmap(ww, xticklabels=xx, yticklabels=xx, linewidths=.5)
        plt.show()
        q = np.array([xx])  # make it 2D
        gt = (q == q.T)
        sns.heatmap(gt, xticklabels=xx, yticklabels=xx, linewidths=.5)
        plt.show()
        a=1
    test_accuracy, confusion_matrix = calculate_accuracy(model, testloader, device)

    print("test accuracy: {:.3f}%".format(test_accuracy))

    # plot confusion matrix
    # fig, ax = plt.subplots(1,1,figsize=(8,6))
    # ax.matshow(confusion_matrix, aspect='auto', vmin=0, vmax=1000, cmap=plt.get_cmap('Blues'))
    # plt.ylabel('Actual Category')
    # plt.yticks(range(10), classes)
    # plt.xlabel('Predicted Category')
    # plt.xticks(range(10), classes)
    # plt.show()


    trained_model = torch.load('/home/smorandv/ac8_and_aging/checkpoints/cifar_cnn_ckpt.pth')
    a = 1