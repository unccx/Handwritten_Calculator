#~~~5. 模型更新 ~~~

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
def model_update(X, y, model):
    
    from tensorflow.keras.optimizers import RMSprop
    from keras.utils.np_utils import to_categorical
    from keras.preprocessing.image import ImageDataGenerator
      
    y = to_categorical(y, num_classes = 14)
    
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(X)

    #freezing layers 0 to 4
    for l in range(0, 5):
        model.layers[l].trainable = False

    optimizer = RMSprop(lr = 0.0001, rho = 0.9, epsilon = 1e-08, decay=0.0 )
    model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])
        
    history = model.fit(
                        datagen.flow(X,y, batch_size = 1),
                        epochs = 50,
                        verbose = 1
                        )
    
    # 保存模型
    model.save("updated_model.h5") 
    
    print("模型已更新!!")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++


# 接受用户对预测的反馈
feedback = input("表达式是否预测正确？ (y/n): ")

# 如果预测不正确，则要求用户提供正确的表达方式
if feedback == "n":
    corr_ans_str = str(input("正确的表达式是: "))
    corr_ans_str = corr_ans_str.replace(" ", "")
    
    def feedback_conversion(correct_ans_str):
        return [char for char in correct_ans_str]
    
    corr_ans_list = feedback_conversion(corr_ans_str)
    dic = {"/":"10", "+": "11", "-": "12", "*": "13"}  
    corr_ans_list = [dic.get(n, n) for n in corr_ans_list]
    corr_ans_arr= np.array(list(map(int, corr_ans_list)))
    print(corr_ans_arr.shape)
    
    # 比较表达式，得到错误的预测元素的索引
    wrong_pred_indices = []
    
    for i in range(len(corr_ans_arr)):
        if corr_ans_arr[i] == elements_pred[i]:
            pass
        else:
            wrong_pred_indices.append(i)
    
    # 获取错误预测的元素
    X = elements_array[[wrong_pred_indices]]

    # 重塑以适应模型输入标准    
    if len(X.shape) == 3:
        X = X.reshape(-1, 28, 28, 1)
    else:
        pass
    
    # 把正确答案作为标签
    y = corr_ans_arr[[wrong_pred_indices]]
    
    # 更新模型
    model_update(X, y, model)    
    
    
else: # 如果表达式预测正确
    pass