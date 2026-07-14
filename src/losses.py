import tensorflow as tf
from keras.saving import register_keras_serializable

# ==========================================
# 1. Base Multiclass Dice Loss
# ==========================================

@register_keras_serializable(package='CustomLosses')
class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes=3, smooth=1e-6, name='dice_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.smooth = smooth

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: Ground truth labels, shape (batch_size, H, W, 1) or (batch_size, H, W)
            y_pred: Network predictions (logits or softmax), shape (batch_size, H, W, num_classes)
        """
        # 1. Ensure y_true is squeezed if it has a trailing channel dimension
        y_true = tf.squeeze(y_true, axis=-1)
        y_true = tf.cast(y_true, tf.int32)
        
        # 2. One-hot encode y_true to match y_pred's shape -> (batch_size, H, W, num_classes)
        y_true_one_hot = tf.one_hot(y_true, depth=self.num_classes, dtype=tf.float32)
        
        # 3. Flatten the spatial dimensions (H, W) to make summing easier
        # Shape becomes: (batch_size, H*W, num_classes)
        y_true_flat = tf.reshape(y_true_one_hot, [tf.shape(y_true_one_hot)[0], -1, self.num_classes])
        y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1, self.num_classes])
        
        # 4. Calculate intersection and denominators per class, averaged across the batch
        # Sum over the spatial dimension (axis=1)
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)
        denominator = tf.reduce_sum(y_true_flat, axis=1) + tf.reduce_sum(y_pred_flat, axis=1)
        
        # 5. Compute Dice score per class per batch element
        dice_per_class = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        
        # 6. Average over all classes and batch elements
        dice_per_image = tf.reduce_mean(dice_per_class, axis=-1)
        mean_dice = tf.reduce_mean(dice_per_image)
        
        # Return loss (1 - Dice)
        return 1.0 - mean_dice

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "smooth": self.smooth,
        })
        return config

# ==========================================
# 2. Combined Dice + CCE Loss
# ==========================================

@register_keras_serializable(package="CustomLosses")
class CombinedDiceCCELoss(tf.keras.losses.Loss):
    def __init__(self, num_classes=3, dice_weight=1.0, cce_weight=1.0, name="combined_loss", **kwargs):

        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.cce_weight = cce_weight
        
        self.dice_loss_fn = DiceLoss(num_classes=num_classes, **kwargs)
        self.cce_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, **kwargs)

    def call(self, y_true, y_pred):
        dice_loss = self.dice_loss_fn(y_true, y_pred)
        cce_loss = self.cce_loss_fn(y_true, y_pred)
        return (self.dice_weight * dice_loss) + (self.cce_weight * cce_loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "dice_weight": self.dice_weight,
            "cce_weight": self.cce_weight,
        })
        return config