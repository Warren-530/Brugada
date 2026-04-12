import os

import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Layer, Reshape

# Global model cache
MODELS = {}

FEATURE_LAYER_BY_MODEL = {
    "resnet": "nature_resnet_feature",
    "blstm": "bilstm_feature",
    "eegnet": "eegnet_feature",
    "cwt_cnn": "cwt_feature",
}

# Notebook-aligned deployment thresholds.
DECISION_THRESHOLD = 0.050
UPPER_BOUND = 0.060
DISPLAY_THRESHOLD = 0.350


@keras.utils.register_keras_serializable()
class LeadSpatialAttention(Layer):
    def __init__(self, **kwargs):
        super(LeadSpatialAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_leads = input_shape[-1]
        self.dense1 = Dense(self.num_leads // 2, activation="relu")
        self.dense2 = Dense(self.num_leads, activation="sigmoid")
        super(LeadSpatialAttention, self).build(input_shape)

    def call(self, inputs):
        squeeze = tf.reduce_mean(inputs, axis=1)
        excitation = self.dense1(squeeze)
        attention_weights = self.dense2(excitation)
        attention_weights = Reshape((1, self.num_leads))(attention_weights)
        return inputs * attention_weights


def load_all_models():
    """Load all .keras and .pkl files centrally."""
    if MODELS:
        return

    print("Initializing Core Models...")
    custom_objs = {"LeadSpatialAttention": LeadSpatialAttention}

    model_files = {
        "resnet": os.path.join("models", "extractor_resnet.keras"),
        "blstm": os.path.join("models", "extractor_bilstm.keras"),
        "eegnet": os.path.join("models", "extractor_eegnet.keras"),
        "cwt_cnn": os.path.join("models", "extractor_cwt_cnn.keras"),
    }

    for model_key, model_file in model_files.items():
        try:
            print(f"  Loading {model_file}...")
            MODELS[model_key] = keras.models.load_model(model_file, custom_objects=custom_objs)
            print(f"    [OK] {model_file} loaded successfully")
        except Exception as e:
            print(f"    [ERROR] Error loading {model_file}: {str(e)[:100]}")
            raise ValueError(
                f"\nFailed to load {model_file}.\n"
                f"This typically means TensorFlow version is incompatible.\n"
                f"Please reinstall TensorFlow 2.12.0:\n"
                f"  pip uninstall tensorflow -y\n"
                f"  pip install tensorflow==2.12.0\n"
                f"Then restart the app."
            )

    try:
        print("  Building feature extraction models...")
        MODELS["resnet_feat"] = keras.Model(
            inputs=MODELS["resnet"].input,
            outputs=MODELS["resnet"].get_layer(FEATURE_LAYER_BY_MODEL["resnet"]).output,
        )
        MODELS["blstm_feat"] = keras.Model(
            inputs=MODELS["blstm"].input,
            outputs=MODELS["blstm"].get_layer(FEATURE_LAYER_BY_MODEL["blstm"]).output,
        )
        MODELS["eegnet_feat"] = keras.Model(
            inputs=MODELS["eegnet"].input,
            outputs=MODELS["eegnet"].get_layer(FEATURE_LAYER_BY_MODEL["eegnet"]).output,
        )
        MODELS["cwt_feat"] = keras.Model(
            inputs=MODELS["cwt_cnn"].input,
            outputs=MODELS["cwt_cnn"].get_layer(FEATURE_LAYER_BY_MODEL["cwt_cnn"]).output,
        )
        print("    [OK] Feature models built successfully")
    except Exception as e:
        print(f"    [ERROR] Error building feature models: {str(e)}")
        raise

    try:
        print("  Loading classifier models...")
        MODELS["scaler"] = joblib.load(os.path.join("models", "brugada_scaler.pkl"))
        MODELS["selector"] = joblib.load(os.path.join("models", "brugada_selector.pkl"))
        MODELS["meta"] = joblib.load(os.path.join("models", "brugada_meta_learner.pkl"))
        print("    [OK] Scaler, selector, and meta-learner loaded successfully")
    except Exception as e:
        print(f"    [ERROR] Error loading classifier models: {str(e)}")
        raise

    print("[OK] All models initialized successfully!")
