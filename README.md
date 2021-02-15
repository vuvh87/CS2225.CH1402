- Nguyễn Hữu Thái - CH1901030
- Võ Hoàng Vũ - CH1902039
- Nguyễn Thúc Hảo - CH1903002


# Tên đề tài: NHẬN DIỆN VI PHẠM QUY ĐỊNH GIÃN CÁCH TRONG DỊCH COVID-19


## Tạo dataset từ thư mục ảnh


```python
train_dir = os.path.join('dataset', 'train')
validation_dir = os.path.join('dataset', 'validate')

BATCH_SIZE = 5
IMG_SIZE = (299, 299)

train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

validation_dataset = image_dataset_from_directory(validation_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)
```

## (Optional) Xem ảnh mẫu trong bộ dữ liệu


```python
class_names = train_dataset.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(len(images)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
```

## Tạo dữ liệu test

Trong bộ dữ liệu không có dữ liệu test vì vậy chuyển 20% dữ liệu train làm dữ liệu test bằng công cụ ```tf.data.experimental.cardinality```.


```python
val_batches = tf.data.experimental.cardinality(train_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)
print('Number of train batches: %d' % tf.data.experimental.cardinality(train_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))
```

### Cấu hình bộ dữ liệu để tăng hiệu suất xử lý

Sử dụng cơ chế nạp bộ đệm để tải hình ảnh từ đĩa để không làm block I/O. Để tìm hiểu thêm về phương pháp này, hãy xem hướng dẫn [hiệu suất xử lý dữ liệu](https://www.tensorflow.org/guide/data_performance).


```python
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
```

### Tăng dữ liệu

Khi bạn không có tập dữ liệu hình ảnh lớn, có thể tăng tính đa dạng mẫu bằng cách áp dụng các phép biến đổi ngẫu nhiên cho hình ảnh train, chẳng hạn như xoay và lật ngang. Điều này giúp mô hình hiển thị các khía cạnh khác nhau của dữ liệu train và giảm [overfitting](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit). Bạn có thể tìm hiểu thêm về tăng dữ liệu trong [hướng dẫn](https://www.tensorflow.org/tutorials/images/data_augmentation) này.


```python
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])
```

Lưu ý: Các lớp này chỉ hoạt động trong quá trình đào tạo. Chúng không được kích hoạt khi mô hình được sử dụng ở chế độ suy luận trong `model.evaluate` hoặc` model.fit`.


```python
for images, _ in train_dataset.take(1):
  plt.figure(figsize=(10, 10))
  first_image = images[0]
  for i in range(len(images)):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
    plt.imshow(augmented_image[0] / 255)
    plt.axis('off')
```


```python
preprocess_input = tf.keras.applications.inception_v3.preprocess_input
```

Lưu ý: Nếu sử dụng `tf.keras.application` khác, hãy nhớ kiểm tra tài liệu API để xác định xem chúng có chấp nhận các pixel trong `[-1,1]` hoặc `[0,1]` hay không, hoặc sử dụng hàm `preprocess_input` đi kèm.

## Tạo base model từ pre-trained convnets

Tạo base model từ mô hình **Inception V3** được phát triển tại Google. Mô hình này đã được pre-trained trên tập dữ liệu ImageNet, một tập dữ liệu lớn bao gồm 1,4 triệu hình ảnh và 1000 lớp. ImageNet là một tập dữ liệu đào tạo dùng trong nghiên cứu với nhiều loại như `jackfruit` và `syringe`. Những kiến ​​thức cơ bản này sẽ giúp chúng phân loại positive và negative từ tập dữ liệu cụ thể sẽ sử dụng.

Đầu tiên, cần chọn lớp Inception V3 sẽ sử dụng để trích xuất tính năng. Lớp phân loại cuối cùng (ở "trên cùng", vì hầu hết các sơ đồ của mô hình học máy đều đi từ dưới lên trên) không hữu ích lắm. Thay vào đó, bạn sẽ làm theo thông lệ phổ biến là phụ thuộc vào lớp cuối cùng trước khi thực hiện thao tác làm phẳng. Lớp này được gọi là "lớp nút cổ chai". Các tính năng của lớp nút cổ chai giữ được tính tổng quát hơn so với lớp cuối cùng / trên cùng.

Đầu tiên, khởi tạo mô hình Inception V3 với các trọng số được đào tạo trên ImageNet. Bằng cách chỉ định đối số **include_top=False**, bạn tải một mạng không bao gồm các lớp phân loại ở trên cùng, lý tưởng cho việc trích xuất đặc trưng.


```python
# Create the base model from the pre-trained model Inception V3
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
```

Bộ trích xuất đặc trưng này chuyển đổi từng hình ảnh `299x299x3` thành một khối đặc trưng `8x8x2048`. Hãy xem những gì nó làm với một loạt hình ảnh ví dụ:


```python
image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)
```

## Trích xuất đặc trưng

Trong bước này, tiến hành đóng băng convolutional base được tạo từ bước trước và sử dụng làm trình trích xuất tính năng. Ngoài ra, thêm một bộ phân loại lên trên nó và train bộ phân loại cấp cao nhất.

### Đóng băng convolutional base

Điều quan trọng là phải đóng băng convolutional base trước khi bạn biên dịch và train mô hình. Việc đóng băng (bằng cách thiết lập layer.trainable = False) ngăn không cho các trọng số trong một lớp nhất định được cập nhật trong quá trình training. Inception V3 có nhiều lớp, vì vậy việc đặt cờ `trainable` của toàn bộ mô hình thành False sẽ đóng băng tất cả chúng.


```python
base_model.trainable = False
```

### Lưu ý quan trọng về các lớp BatchNormalization

Nhiều mô hình chứa các lớp `tf.keras.layers.BatchNormalization`. Lớp này là một trường hợp đặc biệt và nên thực hiện các biện pháp phòng ngừa trong bối cảnh tinh chỉnh, như được trình bày sau trong hướng dẫn này.

Khi bạn đặt `layer.trainable = False`, lớp `BatchNormalization` sẽ chạy ở chế độ suy luận và sẽ không cập nhật trung bình và phương sai của nó.

Khi bạn giải phóng một mô hình có chứa các lớp BatchNormalization để thực hiện tinh chỉnh, bạn nên giữ các lớp BatchNormalization ở chế độ suy luận bằng cách chuyển `training = False` khi gọi mô hình cơ sở. Nếu không, các cập nhật được áp dụng cho các trọng số không thể đào tạo sẽ phá hủy những gì mà mô hình đã học được.

Để biết chi tiết, hãy xem [Hướng dẫn học chuyển tiếp](https://www.tensorflow.org/guide/keras/transfer_learning).


```python
# (Optional) Let's take a look at the base model architecture
base_model.summary()
```

### Thêm đầu phân loại

Để tạo dự đoán từ các khối đặc trưng, hãy tính trung bình trên các vị trí không gian `5x5`, sử dụng lớp `tf.keras.layers.GlobalAveragePooling2D` để chuyển đổi các đặc trưng thành một vectơ 1280 phần tử duy nhất trên mỗi hình ảnh.


```python
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)
```

Áp dụng lớp `tf.keras.layers.Dense` để chuyển đổi các đặc trưng này thành một dự đoán duy nhất cho mỗi hình ảnh. Không cần hàm kích hoạt ở đây vì dự đoán này sẽ được coi là `logit` hoặc giá trị dự đoán thô. Số dương dự đoán hạng 1, số âm dự đoán hạng 0.


```python
prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)
```

Xây dựng mô hình bằng cách xâu chuỗi các lớp tăng dữ liệu, thay đổi tỷ lệ, base_model và trình trích xuất đặc trưng lại với nhau bằng cách sử dụng [Keras Functional API](https://www.tensorflow.org/guide/keras/functional). Như đã đề cập trước đó, hãy sử dụng training=False vì mô hình chứa một lớp BatchNormalization.



```python
inputs = tf.keras.Input(shape=(299, 299, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)
```

### Biên dịch mô hình

Biên dịch mô hình trước khi đào tạo nó. Vì có hai lớp, sử dụng suy hao cross-entropy nhị phân với `from_logits=True` vì mô hình cung cấp đầu ra tuyến tính.


```python
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
```


```python
# (Optional) Preview model
model.summary()
```

2,5tr tham số trong Inception được đóng băng, nhưng có 1,2K _trainable_ tham số trong Dense layer. Chúng được phân chia giữa hai đối tượng `tf.Variable`, trọng số và độ lệch.


```python
len(model.trainable_variables)
```

### Train mô hình

Sau khi train 10 epochs, độ chính xác ~ 87% trên validation set.



```python
# initial_epochs = 10
initial_epochs = 50

loss0, accuracy0 = model.evaluate(validation_dataset)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)
```

### Learning curves

Xem xét các learning curves về độ chính xác / suy hao của quá trình training và valiation khi sử dụng mô hình cơ sở Inception V3 làm trình trích xuất đặc trưng.


```python
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
```

Lưu ý: Nếu thắc mắc tại sao số liệu validation rõ ràng tốt hơn số liệu training, thì yếu tố chính là do các lớp như `tf.keras.layers.BatchNormalization` và `tf.keras.layers.Dropout` ảnh hưởng đến độ chính xác trong quá trình training. Chúng bị tắt khi tính toán suy hao trong validation.

## Fine tuning

Trong quá trình trích xuất đặc trưng, chỉ training một vài lớp trên đầu của mô hình cơ sở Inception V3. Trọng số của pre-trained network **không** được cập nhật trong quá trình training.

Một cách để tăng hiệu suất hơn nữa là train (hoặc "fine tuning") trọng số của các lớp trên cùng của mô hình pre-trained cùng với việc training bộ phân loại đã thêm. Quá trình training sẽ buộc các trọng số được điều chỉnh từ đặc trưng chung ánh xạ sang đặc trưng được liên kết cụ thể với tập dữ liệu.

Lưu ý: Điều này chỉ nên được thực hiện sau khi đã train bộ phân loại cấp cao nhất với mô hình pre-trained đã được đặt thành non-trainable. Nếu thêm một bộ phân loại được khởi tạo ngẫu nhiên lên đầu một mô hình pre-trained và cố gắng train tất cả các lớp cùng nhau, độ lớn của các cập nhật gradient sẽ quá lớn (do trọng số ngẫu nhiên từ bộ phân loại) và mô hình pre-trained của sẽ quên những gì nó đã học.

Ngoài ra, nên cố gắng tinh chỉnh một số lượng nhỏ các lớp trên cùng thay vì toàn bộ mô hình Inception. Trong hầu hết các mạng phức hợp, lớp càng cao thì càng chuyên biệt. Một vài lớp đầu tiên học các tính năng rất đơn giản và chung chung, khái quát cho hầu hết các loại hình ảnh. Khi lên cao hơn, các tính năng ngày càng cụ thể hơn đối với tập dữ liệu mà mô hình được đào tạo. Mục tiêu của việc tinh chỉnh là để điều chỉnh các tính năng chuyên biệt này để hoạt động với tập dữ liệu mới, thay vì ghi đè lên tổng quát.

### Giải phóng các lớp trên cùng của mô hình

Tất cả những gì cần làm là giải phóng `base_model` và đặt các lớp dưới cùng là un-trainable. Sau đó, nên biên dịch lại mô hình (cần thiết để những thay đổi này có hiệu lực) và tiếp tục training.


```python
base_model.trainable = True
```


```python
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
# fine_tune_at = 280
fine_tune_at = 200

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False
```

### Biên dịch mô hình

Khi bạn đang traing một mô hình lớn hơn nhiều và muốn đọc các trọng số đã được huấn luyện trước, điều quan trọng là phải sử dụng tỷ lệ học tập thấp hơn ở giai đoạn này. Nếu không, mô hình của bạn có thể bị overfit rất nhanh.


```python
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])
```

### Tiếp tục train mô hình

Nếu train để hội tụ sớm hơn, bước này sẽ cải thiện độ chính xác lên một vài phần trăm.


```python
# fine_tune_epochs = 5
fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset)
```

Hãy xem xét các learning curves về độ chính xác / suy hao của quá trình training và validation khi tinh chỉnh vài lớp cuối cùng của mô hình cơ sở Inception V3 và training bộ phân loại trên đó. Suy hao trong validation cao hơn nhiều so với suy hao train, vì vậy có thể nhận được một số overfitting.

Sau khi tinh chỉnh, mô hình gần như đạt đến 98% độ chính xác trên validation set.


```python
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']
```


```python
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
```

### Đánh giá và dự đoán

Cuối cùng, có thể xác minh hiệu suất của mô hình trên dữ liệu mới bằng cách sử dụng bộ thử nghiệm.


```python
loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)
```

Và mô hình đã sẵn sàng sử dụng để dự đoán xem ảnh là negative hay positive.


```python
#Retrieve a batch of images from the test set
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

plt.figure(figsize=(10, 10))
for i in range(5):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(class_names[predictions[i]])
  plt.axis("off")
```
