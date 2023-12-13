from keras.layers import *

class Block:
    def __init__(self, nb_filters, kernel_size):
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.end_output = [] #to retrieve the encoder outputs and use them in the concatenation part of the decoder

    #ResNet block configuration method with skip connection establishment
    def resnet_block(self, input_layer):
        x = Conv2D(self.nb_filters, self.kernel_size, padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(self.nb_filters, self.kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = concatenate([input_layer, x], axis=-1)
        x = Activation('relu')(x)
        return x

    #Method of constructing the encoder type blocks for our ResNet Blocks modelling model
    def build_block_encoder_resnet(self, inputs, pooling=True, dropout=False, is_last_layer=False):
        conv1 = Conv2D(self.nb_filters, self.kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(self.nb_filters, self.kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        conv_enc = conv1 #in order to memorize the result of this networks and memorize it for the concatenation
        if dropout :
            conv1 = Dropout(0.5)(conv1)
            conv_enc = conv1
        conv1 = self.resnet_block(conv1)
        if pooling :
            conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        if is_last_layer :
            self.nb_filters /=2
        else :
            self.nb_filters *= 2
            self.end_output.append(conv_enc)
        return conv1
    
    #Method of constructing the standard blocks of our ResNet Blocks model decoder
    def build_block_decoder_resnet(self, inputs, is_last_layer = False):
        up6 = Conv2D(self.nb_filters, self.kernel_size-1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(inputs))
        merge6 = concatenate([self.end_output.pop(), up6], axis = 3) #concatenation between the first element of the decoder and the last element of the encoder
        conv6 = Conv2D(self.nb_filters, self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(self.nb_filters, self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
        conv6 = self.resnet_block(conv6)
        if is_last_layer :
            conv6 = Conv2D(2, self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
            conv6 = Conv2D(1, 1, activation = 'sigmoid')(conv6)
        self.nb_filters /= 2
        return conv6
    

    #Construction method for the encoder blocks of our Conv2D modelling model
    def build_block_encoder_conv2D(self, inputs, pooling = True, dropout = False, is_last_layer = False):
        conv1 = Conv2D(self.nb_filters, self.kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(self.nb_filters, self.kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        conv_enc = conv1
        if dropout :
            conv1 = Dropout(0.5)(conv1)
            conv_enc = conv1
        if pooling :
            conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        if is_last_layer :
            self.nb_filters /=2
        else :
            self.nb_filters *= 2
            self.end_output.append(conv_enc)
        
        return conv1
    
    #Construction method for the decoder blocks of our Conv2D modelling model
    def build_block_decoder_conv2D(self, inputs, is_last_layer = False):
        up6 = Conv2D(self.nb_filters, self.kernel_size-1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(inputs))
        merge6 = concatenate([self.end_output.pop(), up6], axis = 3) #concatenation between the first element of the decoder and the last element of the encoder
        conv6 = Conv2D(self.nb_filters, self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(self.nb_filters, self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
        if is_last_layer :
            conv6 = Conv2D(2, self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
            conv6 = Conv2D(1, 1, activation = 'sigmoid')(conv6)
        self.nb_filters /= 2
        return conv6
    
    def get_filters(self):
        return self.nb_filters
    
    def get_kernel_size(self):
        return self.kernel_size
    
    def get_outputs(self):
        return self.outputs
    
    def get_inputs(self): 
        return self.inputs
    
    def get_end_output(self):
        return self.end_output
    

class Encoder(Block):
    def __init__(self, nb_filters, kernel_size, input_size = (256,256,1), nb_blocks = 3):
        super().__init__(nb_filters, kernel_size)
        self.inputs = Input(input_size)
        self.nb_blocks = nb_blocks
        self.outputs = self.build_model()

    def build_model(self):
        conv = self.inputs #only in order to use the loop
        for i in range(self.nb_blocks):
            conv = self.build_block_encoder_conv2D(conv, pooling = True, dropout = False)

        output_enc = self.build_block_encoder_conv2D(conv, pooling = True, dropout=True)

        #Not really the encoder part, we can call it the middle part between encoder and decoder but because of the structure, we put it in the encoder part
        outputs = self.build_block_encoder_conv2D(output_enc, dropout=True, pooling = False, is_last_layer= True)

        return outputs
    
    
    
class Decoder(Block):
    def __init__(self, nb_blocks, encoder : Encoder): 
        #A decoder cannot construct without his encoder
        self.kernel_size = encoder.get_kernel_size()
        self.nb_filters = encoder.get_filters()
        self.end_output = encoder.get_end_output()
        self.nb_blocks = nb_blocks
        self.encoder = encoder
        self.inputs = encoder.get_outputs()
        self.outputs = self.build_model()

    def build_model(self):
        conv = self.inputs #only in order to use the loop
        for i in range(self.nb_blocks):
            conv = self.build_block_decoder_conv2D(conv)

        #Here, we choose to put a ResNet Block in our architecture, but you can easily change other layer by changing the method you use !
        outputs = self.build_block_decoder_conv2D(conv, is_last_layer = True)
        return outputs
    



    
