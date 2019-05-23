import matplotlib.pyplot as plt


def plot2d(list_of_values, save_path=None):
    plt.plot(list_of_values)

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot2d_multiVal(nested_list_of_values,
                    save_path=None):

    for List in nested_list_of_values:
        plt.plot(List)

    plt.legend(['transformer loss',
                'infersent loss',
                'total loss'], loc=1)

    if save_path:
        plt.savefig(save_path)

    plt.show()


if __name__ == "__main__":
    fpath = "logs/smgsv_ifs_e10/train.log"
    savepath = "logs/smgsv_ifs_e10/trainLoss_ifs.png"
    accu_saveapth = "logs/smgsv_ifs_e10/accuracy_train_ifs.png"

    with open(fpath) as f:
        content = f.readlines()

    lineLength = len(content)
    print(lineLength)

    # print(content[0:100])

    count = 0
    trsLoss = []
    ifsLoss = []
    ttLoss = []
    Accu = []

    # for Original use
    # for item in content:
    #     count += 1
    #     if 'loss:' in item:
    #         loss = item.split(' ')[1]
    #         Loss.append(float(loss))
    #     if 'accuracy' in item:
    #         accu = item.split(',')[1].split(':')
    #         print(len(accu))
    #         if len(accu) == 2:
    #             accu = accu[1].split(' ')[1]
    #             print(accu)
    #             Accu.append(float(accu))

    # for ifs|train use
    for item in content:
        count += 1
        if 'loss:' in item:
            if 'accuracy' not in item:
                loss = item.split(' ')
                # print(loss)
                trsloss = loss[1]
                ifsloss = loss[3]
                ttloss = loss[6]

                trsLoss.append(float(trsloss))
                ifsLoss.append(float(ifsloss))
                ttLoss.append(float(ttloss))

        if 'accuracy' in item:
            if 'Epoch' not in item:  # case of Train
                accu = item.split(',')[1].split(':')
                if len(accu) == 2:
                    accu = accu[1].split(' ')[1]
                    # print(accu)
                    Accu.append(float(accu))

            if 'Epoch' in item:  # case of Valid
                accu = item.split(',')[3].split(':')
                if len(accu) == 2:
                    accu = accu[1].split(' ')[1]
                    print(accu)
                    Accu.append(float(accu))

    valuePack = [trsLoss, ifsLoss, ttLoss]
    plot2d_multiVal(valuePack, savepath)
    plot2d(Accu, accu_saveapth)
