from mnist import MNIST

mndata = MNIST('../resources')

# Loading data into lists.
training_images, training_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

# Writing space separated pixel intensity values of training images into
# a text file, each image in separate line.
with open('../resources/training_images.txt', 'w') as file:
  for image in training_images:
    for pixel in image:
      file.write('%s ' % pixel)
    file.write('\n')

# Writing training images labels to text file, each label in separate line.
with open('../resources/training_labels.txt', 'w') as file:
  for label in training_labels:
    file.write('%s\n' % label)

# Writing space separated pixel intensity values of test images into
# a text file, each image in separate line.
with open('../resources/test_images.txt', 'w') as file:
  for image in test_images:
    for pixel in image:
      file.write('%s ' % pixel)
  file.write('\n')

# Writing test images labels to text file, each label in separate line.
with open('../resources/test_labels.txt', 'w') as file:
  for label in test_labels:
    file.write('%s\n' % label)
