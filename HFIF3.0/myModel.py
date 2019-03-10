import myFunc
import numpy as np
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU, add,GRU
from keras import regularizers,backend,activations,models,callbacks

def myLoss(y_true, y_pred):
    return backend.mean(backend.abs((y_pred - y_true)*y_true), axis=-1)

def myMetric(y_true, y_pred):
    return backend.mean(y_pred*y_true, axis=-1)*10

def myActivation(x):
    return activations.relu(x*0.5, alpha=0.00001, max_value=1.0, threshold=-1.0)

def residual_layer(input_block, filters, kernel_size,reg_const=0.0001):
    x = conv_layer(input_block, filters, kernel_size)
    x = Conv2D(filters = filters , kernel_size = kernel_size
    , data_format="channels_first" , padding = 'same'
    , use_bias=False , activation='linear'
    , kernel_regularizer = regularizers.l2(reg_const)
    )(x)
    x = BatchNormalization()(x)
    x = add([input_block, x])
    x = LeakyReLU()(x)
    return (x)

def conv_layer(x, filters, kernel_size,reg_const=0.0001):
    x = Conv2D(filters = filters , kernel_size = kernel_size
    , data_format="channels_first" , padding = 'same'
    , use_bias=False , activation='linear'
    , kernel_regularizer = regularizers.l2(reg_const)
    )(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return (x)

def value_head(x,reg_const=0.0001):
    x = Conv2D(filters = 1 , kernel_size = (1,1)
    , data_format="channels_first" , padding = 'same'
    , use_bias=False , activation='linear'
    , kernel_regularizer = regularizers.l2(reg_const)
    )(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)
    x = Dense(20 , use_bias=False , activation='linear'
        , kernel_regularizer=regularizers.l2(reg_const)
        )(x)
    x = LeakyReLU()(x)
    x = Dense(1 , use_bias=False #, activation='tanh'
        , kernel_regularizer=regularizers.l2(reg_const)
        , name = 'value_head' )(x)
    return (x)

def buildCNNModel(input_dim,nHiddenLayers=6,filters=75,kernel_size=(4,4),opt='nadam'):
    main_input = Input(shape = input_dim, name = 'main_input')
    x = conv_layer(main_input, filters, kernel_size)
    for h in range(nHiddenLayers):
        x = residual_layer(x, filters, kernel_size)
    model = models.Model(inputs=[main_input], outputs=[value_head(x)])
    model.compile(loss=myLoss,optimizer=opt,metrics=[myMetric])
    return model

def buildRNNModel(xShape,nGRU=(1,1),actFlag='tanh',opt='nadam',doRate=0.23):
    model = models.Sequential()
    actFlag=myActivation
    lenGRU=len(nGRU)
    if lenGRU==1:
        model.add(GRU(xShape[1]*nGRU[0],input_shape=xShape,activation=actFlag,
                recurrent_activation=actFlag,dropout=doRate,
                recurrent_dropout=doRate,return_sequences=False))
    else:
        for i in range(lenGRU):
            if i==0:
                model.add(GRU(xShape[1]*nGRU[i],input_shape=xShape,activation=actFlag,
                    recurrent_activation=actFlag,dropout=doRate,
                    recurrent_dropout=doRate,return_sequences=True))
            elif i==lenGRU-1:
                model.add(GRU(xShape[1]*nGRU[i],activation=actFlag,
                    recurrent_activation=actFlag,dropout=doRate,
                    recurrent_dropout=doRate,return_sequences=False))
            else:
                model.add(GRU(xShape[1]*nGRU[i],activation=actFlag,
                    recurrent_activation=actFlag,dropout=doRate,
                    recurrent_dropout=doRate,return_sequences=True))
    model.add(Dense(1))
    model.compile(loss=myLoss,optimizer=opt,metrics=[myMetric])
    return model

def loadModel(modelfile):
    return models.load_model(modelfile,custom_objects={'myLoss': myLoss,
                                'myMetric':myMetric,'myActivation':myActivation})
    
def generateTrainData(xNormData,nDailyData,nx,ny,iy,nIndu,geneR,nRepeat,batchSize):
    xData=[]
    yData=[]
    for nrpt in range(nRepeat*nRepeat):
        r = np.random.permutation(geneR)
        i=0
        for n in r:
            i+=1
            #xData.append(myFunc.toModelInput(xNormData[(n-nx):n,:-len(ny)],(2,nx,nIndu)))
            xData.append(xNormData[(n-nx):n,:-len(ny)])
            yData.append(xNormData[n,iy-len(ny)])
            if i%batchSize==batchSize-1:
                xData=np.array(xData)
                yData=np.array(yData)
                yield (xData,yData)
                xData=[]
                yData=[]
                
def trainRNNModel(model,xNormData,nDailyData,nx,ny,iy,xTest,yTest,nIndu,batchSize=10000,nRepeat=5):
    #print('Start fit RNN Model...')
    geneR=[]
    ndd=nDailyData-ny[-1]-1
    nday=int(xNormData.shape[0]/ndd)
    for i in range(nday):
        for j in range(nx,ndd):
            geneR.append(i*ndd+j)
     #shuffle
    spb=int(len(geneR)/batchSize)
    eStop=callbacks.EarlyStopping(monitor='val_loss',patience=nRepeat,
                                  mode='min', restore_best_weights=True)
    return model.fit_generator(generateTrainData(xNormData,nDailyData,nx,ny,iy,nIndu,geneR,nRepeat,batchSize),
            validation_data=(xTest,yTest),
            steps_per_epoch=spb,
            callbacks=[eStop],epochs=nRepeat*nRepeat).history
                               
def clearBackEnd():
    backend.clear_session()