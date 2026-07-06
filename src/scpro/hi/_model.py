from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HIVAEConfig:
    input_size: int
    dense_layer_size: int
    latent_size: int
    dropout: float = 0.2
    beta: float = 1.0
    epochs: int = 100
    batch_size: int = 16
    save_model: bool = False


class HIVAE:
    """Triplet VAE used by SCPRO-HI.

    The model is imported lazily so `import scpro` does not require TensorFlow.
    """

    def __init__(self, config: HIVAEConfig):
        self.config = config
        self._build_model()

    def _build_model(self) -> None:
        try:
            import tensorflow as tf
            from tensorflow.keras import backend as K
            from tensorflow.keras.layers import BatchNormalization, Concatenate, Dense, Dropout, Input, Lambda
            from tensorflow.keras.losses import mean_squared_error
            from tensorflow.keras.models import Model
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SCPRO-HI VAE requires TensorFlow. Install scpro[hi] and try again."
            ) from exc

        cfg = self.config
        s1_inp = Input(shape=(cfg.input_size,), name="query")
        s2_inp = Input(shape=(cfg.input_size,), name="positive")
        s3_inp = Input(shape=(cfg.input_size,), name="negative")
        inputs = [s1_inp, s2_inp, s3_inp]

        x1 = BatchNormalization()(Dense(cfg.input_size, activation="elu")(s1_inp))
        x2 = BatchNormalization()(Dense(cfg.input_size, activation="elu")(s2_inp))
        x3 = BatchNormalization()(Dense(cfg.input_size, activation="elu")(s3_inp))
        x = Concatenate(axis=-1)([x1, x2, x3])
        x = BatchNormalization()(Dense(cfg.dense_layer_size, activation="elu")(x))

        z_mean = Dense(cfg.latent_size, name="z_mean")(x)
        z_log_sigma = Dense(cfg.latent_size, name="z_log_sigma", kernel_initializer="zeros")(x)

        def sampling(args):
            mean, log_var = args
            batch = K.shape(mean)[0]
            dim = K.int_shape(mean)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return mean + K.exp(0.5 * log_var) * epsilon

        z = Lambda(sampling, output_shape=(cfg.latent_size,), name="z")([z_mean, z_log_sigma])
        self.encoder = Model(inputs, [z_mean, z_log_sigma, z], name="scpro_hi_encoder")

        latent_inputs = Input(shape=(cfg.latent_size,), name="z_sampling")
        y = BatchNormalization()(Dense(cfg.dense_layer_size, activation="elu")(latent_inputs))
        y = Dropout(cfg.dropout)(y)
        y = BatchNormalization()(Dense(cfg.input_size, activation="elu")(y))
        out = Dense(cfg.input_size, name="corrected_features")(y)
        decoder = Model(latent_inputs, out, name="scpro_hi_decoder")

        outputs = decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name="scpro_hi_tvae")

        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
        s1_loss = mean_squared_error(inputs[0], outputs)
        s2_loss = mean_squared_error(inputs[1], outputs)
        s3_loss = mean_squared_error(inputs[2], outputs)
        reconstruction_loss = K.exp(s1_loss + s2_loss - s3_loss)
        vae_loss = K.mean(reconstruction_loss + cfg.beta * kl_loss)
        self.vae.add_loss(vae_loss)
        adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        self.vae.compile(optimizer=adam)

    def train(self, query, positive, negative) -> None:
        cfg = self.config
        self.vae.fit(
            [query, positive, negative],
            query,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            verbose=0,
            shuffle=True,
        )
        if cfg.save_model:
            self.vae.save_weights("scpro_hi_model_weights.h5")

    def predict(self, query, positive, negative):
        return self.vae.predict([query, positive, negative], batch_size=self.config.batch_size, verbose=0)
