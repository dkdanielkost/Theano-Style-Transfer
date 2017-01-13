from vgg19_model import VGG19
import re
import theano
import theano.tensor as T

def RMSprop(cost, params, lr=1, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def generate_img(content, style, white_noise, layer_weights, bias_weights):

	# puts params in [Weights, bias] format for gradient/update step, might turn into function later
	'''
	params=[]

	for key in layer_weights.keys():
	    ln = key.split('_')
	    for b_key in bias_weights.keys():
	        bn = b_key.split('_')
	        if (ln[0]==bn[0]) and (ln[1]==bn[1]):
	        	params += [theano.shared(value=layer_weights[key], borrow=True), 
	        		theano.shared(value=bias_weights[b_key], borrow=True)]

	'''
	
	wn = theano.shared(value=white_noise.reshape(224,224,3), name='wn', borrow=True)

	vgg19 = VGG19(input_image_shape=(1,3,224,224))
	#cont_vgg = VGG19(input_image_shape=content.shape)
	#style_vgg = VGG19(input_image_shape=style.shape)
	#wn_vgg = VGG19(input_image_shape=white_noise.shape)

	get_content_layer = theano.function(inputs=[vgg19.input],
	                                    outputs=vgg19.conv4_2.output)

	get_style_layers = theano.function(inputs=[vgg19.input],
	                                    outputs=[vgg19.conv1_1.output, vgg19.conv2_1.output,
	                                             vgg19.conv3_1.output, vgg19.conv4_1.output,
	                                             vgg19.conv5_1.output])

	#get_params = theano.function(inputs=[vgg19.input], outputs = [vgg19.conv1_1.params])

	######### Stuff above this only needs to be run once ##########
	#sum(np.mean(np.square(L1[i]+L2[i])) for i in range(len(L1)))
	#might be just wn

	contents = T.mean(T.sqr(get_content_layer(wn.get_value())-get_content_layer(content)))

	wn_style = get_style_layers(wn.get_value())
	pnt_style = get_style_layers(style)
	# might need to edit #
	styles = (T.mean(T.sqr(wn_style[0]-pnt_style[0])) + T.mean(T.sqr(wn_style[1]-pnt_style[1])) +
	          T.mean(T.sqr(wn_style[2]-pnt_style[2])) + T.mean(T.sqr(wn_style[3]-pnt_style[3])) +
	          T.mean(T.sqr(wn_style[4]-pnt_style[4])))

	cost = contents + styles

	#updates = RMSprop(cost, params)
	img_grad = T.grad(cost, wn)

	train_model = theano.function([], cost, updates=[(wn,wn-img_grad)])

	train_model()



#def train_wn(train_func):
	#train_func()


'''
import theano.tensor as T

ima = T.dtensor3('ima')
sty = T.dtensor3('sty')
wn = theano.shared(value=white_noise.reshape(224,224,3), name='wn', borrow=True)
params = [wn]

cost = T.mean(T.sqr(wn-ima))+T.mean(T.sqr(wn-sty))

updates = RMSprop(cost, params)

train_model = theano.function([ima, sty], cost, updates=updates)

for i in range(1000):
    train_model(tubingen, starry_night)
plt.imshow(wn.get_value())
'''