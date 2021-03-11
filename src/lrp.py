from os import listdir
from os.path import isfile, join
from src.utility_funcs import *
from src.variables import *
from src.vgg16_classes import imgclasses
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.autograd import Variable
from torchviz import make_dot


# Returns all the files with their attached paths.
def get_all_photos(path):
    return [join(path, f) for f in listdir(path) if isfile(join(path, f))]


# This function takes in an PIL Image, converts it into a tensor then uses the trained cnn to get
# predict the digit in the image.
def predict_image(model, image_tensor):
    cnn_input = Variable(image_tensor)
    cnn_input = cnn_input.to(device)
    output = model(cnn_input)
    output = output.to(device)  # Does the same
    _, expectedVal = torch.max(output.data, 1)
    prediction = torch.topk(torch.nn.functional.softmax(output.data), 5)
    percent, index, percentage = prediction[0].cpu().detach().numpy(), prediction[1].cpu().detach().numpy(), []
    for i in range(0,5):
        percentage.append((index[0][i], percent[0][i]))
    return expectedVal.sum().item(), percentage


# This function takes in a PIL image, converts it to a tensor and predicts what the image
# will be, it then passes the model and img_tensor to the LRP function which returns the
# relevancy of the input at all layers, and presents them.
def preform_lrp(model, image_tensor, output_dir):
    R = e_lrp(model, image_tensor)
    relevance = process_array([("Linear Layer-1", R[6][0]), ("MaxPool-2", R[5][0]), ("Conv2d-2", R[3][0]),
                               ("MaxPool-1", R[2][0]), ("Conv2d-1", R[0][0])])
    #predicted_val, percentage = predict_image(model, image_tensor)
    #percentagestr = process_percentage(percentage)
    #plot_images(image_tensor, relevance, predicted_val, percentagestr, output_dir)


# This function takes in the model of the Network and an image_tensor. This is what
# The LRP algorithm is applied to, it converts all layers to Conv2D then preforms all
# forward passes so we can get the relevancy of the network at all layers.
# This function returns the relevance [R] of all layers within the network (except dropout).
def e_lrp(model, img_tensor):
    #X = (torch.FloatTensor(img_tensor[np.newaxis].transpose([0,3,1,2])*1) - MEAN) / STD
    X = img_tensor
    layers = list(model._modules['features']) + toconv(list(model._modules['classifier']))
    L = len(layers)
    A = [X]+[None]*L
    for l in range(L):
        A[l+1] = layers[l].forward(A[l])
    T = torch.FloatTensor((1.0*(np.arange(1000)==483).reshape([1,1000,1,1])))
    R = [None]*L + [(A[-1]*T).data]
    for l in range(1, L)[::-1]:
        A[l] = (A[l].data).requires_grad_(True)
        if isinstance(layers[l], torch.nn.MaxPool2d):
            layers[l] = torch.nn.AvgPool2d(2)
        if isinstance(layers[l], torch.nn.Conv2d) or isinstance(layers[l],torch.nn.AvgPool2d):
            if l <= 16:
                rho = lambda p: p + 0.25*p.clamp(min=0)
                incr = lambda z: z+1e-9
            if 17 <= l <= 30:
                rho = lambda p: p
                incr = lambda z: z+1e-9+0.25*((z**2).mean()**.5).data
            if l >= 31:
                rho = lambda p: p
                incr = lambda z: z+1e-9

            z = incr(newlayer(layers[l], rho).forward(A[l]))  # step 1
            s = (R[l+1]/z).data                                    # step 2
            (z*s).sum().backward(); c = A[l].grad                  # step 3
            R[l] = (A[l]*c).data                                   # step 4
        else:
            R[l] = R[l+1]

    A[0] = (A[0].data).requires_grad_(True)

    lb = (A[0].data*0+(0-MEAN)/STD).requires_grad_(True)
    hb = (A[0].data*0+(1-MEAN)/STD).requires_grad_(True)

    z = layers[0].forward(A[0]) + 1e-9                                     # step 1 (a)
    z -= newlayer(layers[0],lambda p: p.clamp(min=0)).forward(lb)    # step 1 (b)
    z -= newlayer(layers[0],lambda p: p.clamp(max=0)).forward(hb)    # step 1 (c)
    s = (R[1]/z).data                                                      # step 2
    (z*s).sum().backward(); c,cp,cm = A[0].grad,lb.grad,hb.grad            # step 3
    R[0] = (A[0]*c+lb*cp+hb*cm).data                                       # step 4
    heatmap(np.array(R[0][0]).sum(axis=0),3.5,3.5)


# This function is used to display LRP and the input image with the predicted values
# and the likely-hood the model thinks its correct.
def plot_images(init_img, R, predicted_val, outstring, output_dir):
    fig = plt.figure(figsize=(10, 10))
    columns = 3
    rows = 2
    i = 2  # As we are adding the input image first.
    fig.add_subplot(rows, columns, 1).set_title("Input Image")
    plt.axis('off')
    plt.imshow(init_img.reshape(28, 28), cmap='gray')
    for label, r in R:
        b = 10 * ((np.abs(r) ** 3.0).mean() ** (1.0/3))
        my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
        my_cmap[:, 0:3] *= 0.85
        my_cmap = ListedColormap(my_cmap)
        fig.add_subplot(rows, columns, i).set_title(label)
        plt.axis('off')
        plt.imshow(r, cmap=my_cmap, vmin=-b, vmax=b, interpolation='nearest')
        i = i + 1
    plt.tight_layout()
    plt.figtext(0.5, 0.04, "Predicted value of Network is {} \n {}".format(predicted_val, outstring), ha="center", fontsize=18,
                bbox={"facecolor":"purple", "alpha":0.5, "pad":5})
    fig.savefig(output_dir)


def plot_single_image(R, output_dir, cmap=None):
    b = 10 * ((np.abs(R) ** 3.0).mean() ** (1.0/3))
    if cmap is None:
        my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
        my_cmap[:,0:3] *= 0.85
        my_cmap = ListedColormap(my_cmap)
    else:
        my_cmap = cmap
    fig = plt.figure(figsize=(10, 10))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('off')
    plt.imshow(R, cmap=my_cmap, vmin=-b, vmax=b, interpolation='nearest')
    fig.savefig(output_dir)