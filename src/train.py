from src.model import build_unet
import tensorflow as tf
from src.losses import DiceLoss, CombinedDiceCCELoss
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau 
from src.config import BATCH_SIZE, IMAGE_SIZE, EPOCHS, LEARNING_RATE

# Compile the model with Dice loss and an optimizer

def create_model(learning_rate=LEARNING_RATE):
    """
    Compile the given model with Dice loss and an Adam optimizer.

    Args:
        learning_rate: Learning rate for the optimizer.
    """
    model = build_unet(input_shape=(*IMAGE_SIZE, 3), num_classes=3)
    #dice_loss = DiceLoss(num_classes=3)
    custom_total_loss = CombinedDiceCCELoss(num_classes=3, dice_weight=1.3, cce_weight=0.7)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=custom_total_loss, metrics=['accuracy', tf.keras.metrics.IoU(
        num_classes=3,
        target_class_ids=[0,1,2],
        sparse_y_true=True,
        sparse_y_pred=False,
        name="mean_iou"
    )])
    return model

def create_callbacks(checkpoint_path='models/dice_cce_unet.keras'):
    """
    Create callbacks for early stopping, model checkpointing, and learning rate reduction.

    Args:
        checkpoint_path: Path to save the best model checkpoint.
    """
    early_stop = EarlyStopping(
        monitor='val_loss',       
        patience=5,               
        restore_best_weights=True 
    )

    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path, 
        monitor='val_loss',      
        save_best_only=True,      
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-5,
        verbose=1
    )

    return [early_stop, checkpoint, reduce_lr]

def train_model(model, train_ds, val_ds, epochs=EPOCHS, checkpoint_path='models/dice_cce_unet.keras'):
    """
    Train the given model on the training dataset and validate on the validation dataset.

    Args:
        model: A TensorFlow Keras model instance.
        train_ds: Training dataset.
        val_ds: Validation dataset.
        epochs: Number of epochs to train the model.
        checkpoint_path: Path to save the best model checkpoint.
    """
    callbacks = create_callbacks(checkpoint_path)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    return history
