import tensorflow as tf
import tensorflow_addons as tfa
from os import listdir
import argparse
import os

from BERT_models import build_linear_classifier_model
from load_data import load_data


def main(args):
      X_train, X_validation, _, y_train, y_validation, _, _ = load_data(args.data, args.mode)
      print(f"Training on {len(X_train)} samples\nUsing {len(X_validation)} samples for validation")
      
      classifier_model = build_linear_classifier_model('electra_base', args.mode)
      classifier_model.summary()

      if args.mode == 'binary':
            metrics = [tfa.metrics.F1Score(num_classes = 1, threshold = 0.5, average = "micro", name = 'f1_micro'), 
                       tf.metrics.BinaryAccuracy()]
            
            loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = args.smoothing)

      else:
            metrics = [tfa.metrics.F1Score(num_classes = 2, average = "macro", name = 'f1_macro'), 
                       tfa.metrics.F1Score(num_classes = 2, average = "weighted", name = 'f1_weighted'), 
                       tfa.metrics.F1Score(num_classes = 2, average = "micro", name = 'f1_micro'),
                       tf.metrics.CategoricalAccuracy()]
            
            loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing = args.smoothing)


      optimizer = tf.keras.optimizers.Adam(learning_rate = args.lr)
      classifier_model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

      classifier_model.fit(x = X_train,
                           y = y_train,
                           batch_size = args.batch_size,
                           validation_data = (X_validation, y_validation),
                           epochs = args.epochs)
      model_name = f"model_name"
      save_path = os.path.join("saved_models", model_name)
      classifier_model.save(save_path, include_optimizer=True)



      #classifier_model.save(f'saved_models/0{len(listdir("saved_models")) + 1}', include_optimizer = True)


if __name__ == "__main__":
      parser = argparse.ArgumentParser('Training BERT to perform SUD classification')

      parser.add_argument('--data', type = str, help = 'Path to the dataset')
      parser.add_argument('--mode', type = str, default = 'binary', choices = ['binary', 'multi-class'], help = 'Perform binary or multi-class SUD classification')
      parser.add_argument('--batch_size', type = int, default = 32, help = 'Batch size')
      parser.add_argument('--epochs', type = int, default = 5, help = 'Number of epochs')
      parser.add_argument('--lr', type = float, default = 2e-5, help = 'Learning rate')
      parser.add_argument('--smoothing', type = float, default = 0.1, help = 'Label smoothing rate')
      
      args = parser.parse_args()
      main(args)
