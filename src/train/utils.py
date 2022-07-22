import matplotlib.pyplot as plt


def plot_performance(img_train,
                     img_test,
                     target_train,
                     target_test,
                     output_train,
                     output_test,
                     losses_train,
                     losses_test,
                     figsize=(10, 20)):

    fig, ax = plt.subplots(2, 4, figsize=figsize, gridspec_kw={'width_ratios': [1, 1, 1, 2]})

    ax[0, 0].imshow(img_train)
    ax[0, 0].set_title('Train Input')
    ax[0, 1].imshow(target_train)
    ax[0, 1].set_title('Train Target')
    ax[0, 2].imshow(output_train)
    ax[0, 2].set_title('Train Prediction')
    ax[0, 3].plot(losses_train)
    ax[0, 3].set_title('Train Loss')

    ax[1, 0].imshow(img_test)
    ax[1, 0].set_title('Test Input')
    ax[1, 1].imshow(target_test)
    ax[1, 1].set_title('Test Target')
    ax[1, 2].imshow(output_test)
    ax[1, 2].set_title('Test Prediction')
    ax[1, 3].plot(losses_test)
    ax[1, 3].set_title('Test Loss')

    plt.show()

    # ax1 = plt.subplot(231, figsize=(10, 10))
    # ax1.imshow(input_img)
    # ax1.set_title('Input')
    #
    # ax2 = plt.subplot(232, figsize=(10, 10))
    # ax2.imshow(target)
    # ax2.set_title('Target')
    #
    # ax3 = plt.subplot(233, figsize=(10, 10))
    # ax3.imshow(output)
    # ax3.set_title('Output')
    #
    # ax4 = plt.subplot(212, figsize=(10, 10))
    # ax4.plot(losses)
    # ax4.set_title('Losses')

    #plt.show()
