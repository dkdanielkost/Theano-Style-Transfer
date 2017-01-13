import numpy as np
import theano
import theano.tensor as T
from scipy.optimize import fmin_l_bfgs_b
from scipy import misc
import skimage
from skimage.transform import resize
from skimage.io import imread
from theano_model import vgg19_model, preliminary_model

VGG19 = vgg19_model.VGG19

class NeuralStyleTransfer(object):
    """
    Takes two inputs:
        Content image
        Style image
    Returns:
        A new image with the style of the style input image
        and the content of the content input image
    """
    def __init__(self, content_path, style_path, image_w=500, image_h=500, style_weight=2e5, content_weight=0.001):
        self.image_w = image_w
        self.image_h = image_h
        # mean pixel values for VGG-19
        self.mean_pixels = np.array([104, 117, 123]).reshape((3,1,1))
        content_image = imread(content_path)
        style_image = imread(style_path)
        content = self.preprocess_image(content_image)
        style = self.preprocess_image(style_image)
        self.content = content
        self.style = style

        self.vgg19 = VGG19(input_image_shape=(1,3,image_h,image_w), pool_method = 'average_exc_pad')


        self.get_content_layer = theano.function(inputs=[self.vgg19.input],
                                            outputs=self.vgg19.conv4_2.output)

        self.get_style_layers = theano.function(inputs=[self.vgg19.input],
                                            outputs=[self.vgg19.conv1_1.output, self.vgg19.conv2_1.output,
                                                     self.vgg19.conv3_1.output, self.vgg19.conv4_1.output,
                                                     self.vgg19.conv5_1.output])

        self.sty_out = self.get_style_layers(style)
        self.cont_lay = self.get_content_layer(content)
        white_noise = np.random.uniform(low=-128.0, high=128.0,
         size = (1, 3, image_h,image_w )).astype(theano.config.floatX)
        self.wn = theano.shared(value=white_noise, name='wn', borrow=False)
        self.cont = T.dtensor4('cont')
        self.sty1 = T.dtensor4('sty1')
        self.sty2 = T.dtensor4('sty2')
        self.sty3 = T.dtensor4('sty3')
        self.sty4 = T.dtensor4('sty4')
        self.sty5 = T.dtensor4('sty5')

        # alpha/beta should be around 0.01 to achieve good results
        self.cont_loss = content_weight*0.5*T.sum(T.sqr(self.vgg19.conv4_2.output-self.cont))

        self.style_loss = style_weight*(self.calc_style_loss(self.sty1, self.vgg19.conv1_1.output) +
                     self.calc_style_loss(self.sty2, self.vgg19.conv2_1.output) +
                     self.calc_style_loss(self.sty3, self.vgg19.conv3_1.output) +
                     self.calc_style_loss(self.sty4, self.vgg19.conv4_1.output) +
                     self.calc_style_loss(self.sty5, self.vgg19.conv5_1.output))

        self.cost = self.cont_loss + self.style_loss

        self.img_grad = T.grad(self.cost, self.vgg19.input)

        # Theano functions to evaluate loss and gradient
        self.f_loss = theano.function([self.vgg19.input, self.cont, self.sty1,
         self.sty2, self.sty3, self.sty4, self.sty5], self.cost)
        self.f_grad = theano.function([self.vgg19.input, self.cont, self.sty1,
         self.sty2, self.sty3, self.sty4, self.sty5], self.img_grad)
        
        self.losses = []

    def fit(self, iterations=80, save_every_n=10, optimizer='l-bfgs', alpha=1):
        im = self.wn.get_value().astype('float64')
        self.images = []
        self.images.append(im)

        if optimizer == "l-bfgs":
            for i in range(iterations / save_every_n):
                fmin_l_bfgs_b(self.lbfgs_loss, im.flatten(), fprime=self.lbfgs_gradient, maxfun=save_every_n)
                print("Training @ iteration %s. Cost = %s" % (i * save_every_n, self.losses[-1]))
                im = self.wn.get_value().astype('float64')
                self.images.append(im)
        elif optimizer == "adam":
            adam = Adam(self.wn)
            self.output = adam.optimize(self.cost, self.vgg19.input, alpha=alpha)
            get_output = theano.function([self.vgg19.input, self.cont, self.sty1,
                                          self.sty2, self.sty3, self.sty4, self.sty5]
                                         ,[self.cost, self.output], on_unused_input='ignore')
            for i in range(iterations):
                a_cost, updated_image = get_output(self.wn.get_value(), self.cont_lay, self.sty_out[0],
                                                   self.sty_out[1], self.sty_out[2], self.sty_out[3], self.sty_out[4])
                self.wn.set_value(updated_image)
                self.losses.append(a_cost)
                if i % save_every_n == 0:
                    self.images.append(self.wn.get_value())
                    print("Training @ iteration %s. Cost = %s" % (i, a_cost))
        else:
            print("Oops, your choices are: 'l-bfgs' or 'adam'")

    def preprocess_image(self, im):
        im = skimage.transform.resize(im, (self.image_h, self.image_w), preserve_range=True)
        # Reorder dimensions to have channels first
        im = np.transpose(im,(2,0,1))
        # Convert RGB to BGR (VGG-19 uses BGR)
        im = im[::-1, :, :]
        im = im - self.mean_pixels
        # Add dimension to create a 4D tensor
        im = (im[np.newaxis]).astype(theano.config.floatX)
        return im

    def get_gram_matrix(self, mat):
        mat = mat.flatten(ndim=3)
        g = T.tensordot(mat, mat, axes=([2], [2]))
        return g

    def calc_style_loss(self, style, gen_img):
        sm = T.sum(T.sqr(self.get_gram_matrix(style)-self.get_gram_matrix(gen_img)))
        N = style.shape[1]
        M = style.shape[2]*style.shape[3]
        E = 1.0/(4*N**2*M**2)
        return E*sm

    def lbfgs_loss(self, im):
        # Convert from flattened array from scipy to image dimensions, convert to 32 for theano        
        im = im.reshape((1, 3, self.image_h,self.image_w)).astype(theano.config.floatX)
        # Update wn        
        self.wn.set_value(im)
        # Calculate Gradient and then convert to 64 for scipy        
        loss =  self.f_loss(self.wn.get_value(), self.cont_lay, self.sty_out[0],
         self.sty_out[1], self.sty_out[2], self.sty_out[3],
          self.sty_out[4]).astype('float64')
        self.losses.append(loss)
        return loss

    def lbfgs_gradient(self, im):
        # Convert from flattened array from scipy to image dimensions, convert to 32 for theano
        im = im.reshape((1, 3, self.image_h,self.image_w)).astype(theano.config.floatX)
        # Update wn
        self.wn.set_value(im)
        # Calculate Gradient and then convert to 64 for scipy
        gradient = np.array(self.f_grad(self.wn.get_value(), self.cont_lay,
         self.sty_out[0], self.sty_out[1], self.sty_out[2], self.sty_out[3],
          self.sty_out[4])).flatten().astype('float64')
        return gradient

    def recover_image(self, im):
        im = np.copy(im[0])
        im += self.mean_pixels
        # convert back to RGB
        im = im[::-1]
        # Place channels in 3rd dimension for plotting
        im = np.transpose(im,(1,2,0))
        # Truncate values to be within image values
        im = np.clip(im, 0, 255).astype('uint8')
        return im
    
    def final_image(self):
        return self.recover_image(self.images[-1])

class Adam:
    # https://arxiv.org/pdf/1412.6980v8.pdf
    def __init__(self, param):
        self.m_0 = theano.shared(np.zeros(param.get_value().shape, dtype=theano.config.floatX))
        self.v_0 = theano.shared(np.zeros(param.get_value().shape, dtype=theano.config.floatX))
        self.t_0 = theano.shared(np.float32(0))
        
    def optimize(self, cost, param, alpha=1, beta1=0.9, beta2=0.999, epsilon=10e-8):
        grad = T.grad(cost, param)
        updates = []
        t = self.t_0 + 1
        m_0 = self.m_0
        v_0 = self.v_0
        m = beta1 * m_0 + (1 - beta1) * grad                            
        v = beta2 * v_0 + (1 - beta2) * grad**2 
        m_hat = m / (1-beta1**t)
        v_hat = v / (1-beta2**t)
        param_t = param - (alpha * m_hat) / (T.sqrt(v_hat) + epsilon)
        self.m_0 = m
        self.v_0 = v
        self.t_0 = t
        return param_t